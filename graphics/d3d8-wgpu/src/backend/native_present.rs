use crate::RenderError;
use recomp_gpu::PhysicalLoss;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

pub(crate) struct NativeSlot {
    pub texture: wgpu::Texture,
    pub busy: AtomicU32,
}
pub(crate) struct NativeReadiness {
    slot: Arc<NativeSlot>,
    state: AtomicU32,
    loss: Option<PhysicalLoss>,
}
impl Drop for NativeReadiness {
    fn drop(&mut self) {
        self.slot.busy.store(0, Ordering::Release);
    }
}
struct NativeLease {
    readiness: Arc<NativeReadiness>,
    context: Arc<super::GpuContext>,
}
pub(crate) fn begin_slot(
    slot: Arc<NativeSlot>,
    loss: Option<PhysicalLoss>,
) -> Arc<NativeReadiness> {
    slot.busy.store(1, Ordering::Release);
    Arc::new(NativeReadiness {
        slot,
        state: AtomicU32::new(0),
        loss,
    })
}
pub(crate) struct NativeSubmission {
    readiness: Arc<NativeReadiness>,
    source: wgpu::Texture,
    destination: wgpu::Texture,
    device: wgpu::Device,
    submitted: bool,
}
impl NativeSubmission {
    pub(crate) fn new(
        readiness: Arc<NativeReadiness>,
        source: wgpu::Texture,
        destination: wgpu::Texture,
        device: wgpu::Device,
    ) -> Self {
        Self {
            readiness,
            source,
            destination,
            device,
            submitted: false,
        }
    }
    pub(crate) fn submit(mut self, queue: &wgpu::Queue) {
        if self
            .readiness
            .loss
            .as_ref()
            .is_some_and(PhysicalLoss::is_lost)
        {
            return;
        }
        match submit_copy(
            &self.device,
            queue,
            &self.source,
            &self.destination,
            Arc::clone(&self.readiness),
            None,
        ) {
            Ok(()) => self.submitted = true,
            Err(error) => log::error!("present submission failed: {error}"),
        }
    }
}
impl Drop for NativeSubmission {
    fn drop(&mut self) {
        if !self.submitted {
            self.readiness.state.store(2, Ordering::Release);
        }
    }
}
pub(crate) fn retain_slot(
    readiness: Arc<NativeReadiness>,
    context: Arc<super::GpuContext>,
) -> usize {
    Box::into_raw(Box::new(NativeLease { readiness, context })) as usize
}
pub(crate) unsafe fn lease_texture(
    lease: usize,
) -> Option<(Arc<super::GpuContext>, wgpu::Texture)> {
    let lease = unsafe { (lease as *const NativeLease).as_ref() }?;
    Some((
        Arc::clone(&lease.context),
        lease.readiness.slot.texture.clone(),
    ))
}
pub unsafe fn present_lease_ready(lease: usize) -> u32 {
    let Some(lease) = (unsafe { (lease as *const NativeLease).as_ref() }) else {
        return 2;
    };
    if lease
        .readiness
        .loss
        .as_ref()
        .is_some_and(PhysicalLoss::is_lost)
    {
        return 2;
    }
    let _ = lease.context.device.poll(wgpu::PollType::Poll);
    lease.readiness.state.load(Ordering::Acquire)
}
pub unsafe fn release_present_lease(lease: usize) {
    if lease != 0 {
        drop(unsafe { Box::from_raw(lease as *mut NativeLease) });
    }
}
const CONVERT_RGB565: u32 = 1;
const CONVERT_GAMMA: u32 = 2;
const GAMMA_TABLE_BYTES: u64 = 256 * 16;

struct GammaState {
    serial: u64,
    table: Option<Vec<f32>>,
}

static GAMMA: Mutex<GammaState> = Mutex::new(GammaState {
    serial: 0,
    table: None,
});

/// The guest's display gamma ramp (256 red, then green, then blue entries,
/// 0..65535). Presented frames pass through it on their way to the host, as
/// they would through a display's gamma table. None, or an identity ramp,
/// leaves frames untouched.
pub fn set_gamma_ramp(ramp: Option<&[u16; 768]>) {
    let table = ramp.and_then(|ramp| {
        let identity = (0..768).all(|i| {
            let expected = (i % 256) as f64 / 255.0;
            (f64::from(ramp[i]) / 65535.0 - expected).abs() < 0.5 / 255.0
        });
        (!identity).then(|| {
            let mut table = vec![1.0f32; 1024];
            for i in 0..256 {
                for channel in 0..3 {
                    table[i * 4 + channel] = f32::from(ramp[channel * 256 + i]) / 65535.0;
                }
            }
            table
        })
    });
    let mut state = GAMMA.lock().unwrap_or_else(|e| e.into_inner());
    if state.table != table {
        state.table = table;
        state.serial += 1;
    }
}

pub(crate) fn gamma_active() -> bool {
    GAMMA
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .table
        .is_some()
}

pub(crate) struct NativeConverter {
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    table: wgpu::Buffer,
    gamma_serial: u64,
    gamma: bool,
}
impl NativeConverter {
    /// Brings the gamma table up to date and records what this present needs;
    /// false when a plain copy will do.
    pub(crate) fn prepare(&mut self, queue: &wgpu::Queue, rgb565: bool) -> bool {
        {
            let state = GAMMA.lock().unwrap_or_else(|e| e.into_inner());
            if state.serial != self.gamma_serial {
                if let Some(table) = &state.table {
                    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_ne_bytes()).collect();
                    queue.write_buffer(&self.table, 0, &bytes);
                }
                self.gamma = state.table.is_some();
                self.gamma_serial = state.serial;
            }
        }
        let flags = (if rgb565 { CONVERT_RGB565 } else { 0 })
            | (if self.gamma { CONVERT_GAMMA } else { 0 });
        let mut params = [0u8; 16];
        params[..4].copy_from_slice(&flags.to_ne_bytes());
        queue.write_buffer(&self.params, 0, &params);
        flags != 0
    }
}
pub(crate) fn create_converter(device: &wgpu::Device) -> Result<NativeConverter, RenderError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("present-convert"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct Params { flags: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> table: array<vec4<f32>, 256>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) p: vec3<u32>) {
    let size = textureDimensions(dst);
    if p.x >= size.x || p.y >= size.y { return; }
    let texel = textureLoad(src, vec2<i32>(p.xy), 0);
    var rgb = texel.rgb;
    var alpha = texel.a;
    if (params.flags & 1u) != 0u {
        let bytes = vec3<u32>(rgb * 255.0 + vec3<f32>(0.5));
        let reduced = bytes >> vec3<u32>(3u, 2u, 3u);
        let expanded = reduced * vec3<u32>(255u) / vec3<u32>(31u, 63u, 31u);
        rgb = vec3<f32>(expanded) / 255.0;
        alpha = 1.0;
    }
    if (params.flags & 2u) != 0u {
        let x = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0;
        let lo = vec3<u32>(floor(x));
        let hi = min(lo + vec3<u32>(1u), vec3<u32>(255u));
        let t = x - floor(x);
        rgb = vec3<f32>(mix(table[lo.r].r, table[hi.r].r, t.r),
                        mix(table[lo.g].g, table[hi.g].g, t.g),
                        mix(table[lo.b].b, table[hi.b].b, t.b));
    }
    textureStore(dst, vec2<i32>(p.xy), vec4<f32>(rgb, alpha));
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("present-convert"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("present-convert params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let table = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("present-convert gamma"),
        size: GAMMA_TABLE_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    Ok(NativeConverter {
        pipeline,
        params,
        table,
        gamma_serial: 0,
        gamma: false,
    })
}
pub(crate) fn submit_copy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Texture,
    destination: &wgpu::Texture,
    readiness: Arc<NativeReadiness>,
    converter: Option<&NativeConverter>,
) -> Result<(), RenderError> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("present-copy"),
    });
    if let Some(converter) = converter {
        let src = source.create_view(&Default::default());
        let dst = destination.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present-convert"),
            layout: &converter.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: converter.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: converter.table.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&converter.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(
            destination.width().div_ceil(8),
            destination.height().div_ceil(8),
            1,
        );
    } else {
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            destination.as_image_copy(),
            wgpu::Extent3d {
                width: destination.width(),
                height: destination.height(),
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit([encoder.finish()]);
    queue.on_submitted_work_done(move || {
        readiness.state.store(1, Ordering::Release);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_ramp_is_off_for_identity_and_on_otherwise() {
        let identity: [u16; 768] = std::array::from_fn(|i| ((i % 256) * 257) as u16);
        set_gamma_ramp(Some(&identity));
        assert!(!gamma_active());
        let bright: [u16; 768] = std::array::from_fn(|i| {
            (((i % 256) as f64 / 255.0).powf(0.8) * 65535.0).round() as u16
        });
        set_gamma_ramp(Some(&bright));
        assert!(gamma_active());
        let table = GAMMA.lock().unwrap().table.clone().unwrap();
        assert!(table[128 * 4] > 128.0 / 255.0);
        assert_eq!(table[255 * 4 + 3], 1.0);
        set_gamma_ramp(None);
        assert!(!gamma_active());
    }
}
