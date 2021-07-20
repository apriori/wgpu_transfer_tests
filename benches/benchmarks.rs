#![feature(custom_test_frameworks)]
#![feature(async_closure)]
#![test_runner(criterion::runner)]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use criterion::{Criterion, Throughput};
use criterion_macro::criterion;
use futures::FutureExt;
use scopeguard::ScopeGuard;
use wgpu::{BufferDescriptor, Device, Maintain, MapMode, Queue};

pub fn create_n_mmap_buffers(device: &Arc<Device>, n: usize, size: usize) -> Vec<wgpu::Buffer> {
    (0..n)
        .map(|i| {
            device.create_buffer(&BufferDescriptor {
                label: Some(format!("i{}", i).as_str()),
                size: size as u64,
                usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect()
}

pub async fn get_device() -> (Device, Queue) {
    wgpu_subscriber::initialize_default_subscriber(None);
    let instance = wgpu::Instance::new(wgpu::BackendBit::VULKAN);

    let features = wgpu::Features::PUSH_CONSTANTS;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    (device, queue)
}

#[criterion]
fn transfer_benchmarks(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let (device, queue) = rt.block_on(get_device());

    let arc_device = Arc::new(device);
    let arc_poll_device = arc_device.clone();
    let arc_terminate = Arc::new(Mutex::new(false));
    let arc_terminate_outer = arc_terminate.clone();

    let handle = std::thread::spawn(move || {
        while !*arc_terminate.lock().unwrap() {
            arc_poll_device.poll(Maintain::Wait);
            std::thread::sleep(Duration::from_millis(100));
        }
    });

    let _guard = scopeguard::guard((), |_| {
        *arc_terminate_outer.lock().unwrap() = true;
        handle.join().unwrap();
    });

    const BUFFER_COUNT: usize = 12;
    let size = 2048 * 1536 * 1; // 8 bit 3.2MP image size
    let buffers = create_n_mmap_buffers(&arc_device, BUFFER_COUNT, size);
    let dummy_image = vec![0u8; size];

    {
        let slices_futures = buffers
            .iter()
            .map(|b| {
                let slice = b.slice(..);
                slice.map_async(MapMode::Write).map(move |_| slice)
            })
            .collect::<Vec<_>>();

        let mut slices = rt.block_on(futures::future::join_all(slices_futures));

        let mut transfer_test_group = c.benchmark_group("transfer_test_group");
        transfer_test_group.throughput(Throughput::Bytes((size * BUFFER_COUNT) as u64));
        transfer_test_group.bench_function("12 buffers (premapped)", |b| {
            b.iter(|| {
                for i in 0..BUFFER_COUNT {
                    let slice = &mut slices[i];
                    {
                        let mut view = slice.get_mapped_range_mut();
                        view.copy_from_slice(&dummy_image);
                    }
                }
            })
        });
        // unmap everything. this might be fooling us in the benchmark
        for buf in &buffers {
            buf.unmap();
        }
    }
    {
        let mut transfer_test_group_slow = c.benchmark_group("transfer_test_group_slow");
        transfer_test_group_slow.sample_size(10);
        transfer_test_group_slow.throughput(Throughput::Bytes((size * BUFFER_COUNT) as u64));
        transfer_test_group_slow.bench_function("12 buffers (including map & unmap)", |b| {
            b.iter(|| {
                for i in 0..BUFFER_COUNT {
                    let b = &buffers[i];

                    let slice = b.slice(..);

                    rt.block_on(slice.map_async(MapMode::Write).map(|x| {
                        if x.is_ok() {
                            {
                                let mut view = slice.get_mapped_range_mut();
                                view.copy_from_slice(&dummy_image);
                            }
                            b.unmap();
                        }
                    }));
                }
            })
        });
    }
    {
        let mut transfer_test_group_slow = c.benchmark_group("transfer_test_group_async");
        transfer_test_group_slow.sample_size(10);
        transfer_test_group_slow.throughput(Throughput::Bytes((size * BUFFER_COUNT) as u64));
        transfer_test_group_slow.bench_function(
            "12 buffers (including map & unmap & async)",
            move |b| {
                let arc_buffers = &Arc::new(&buffers);
                let seperated_images = (0..BUFFER_COUNT).map(|_| dummy_image.clone()).collect::<Vec<_>>();
                let arc_images = &Arc::new(seperated_images);

                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(async move || {
                        let futures = futures::stream::FuturesUnordered::new();

                        for (i, buf) in arc_buffers.iter().enumerate() {
                            let slice = buf.slice(..);

                            futures.push(slice.map_async(MapMode::Write).map(move |x| {
                                if x.is_ok() {
                                    {
                                        let mut view = slice.get_mapped_range_mut();
                                        view.copy_from_slice(&arc_images[i]);
                                    }
                                    buf.unmap();
                                }
                            }));
                        }
                        futures::future::join_all(futures).await;
                    })
            },
        );
    }
}
