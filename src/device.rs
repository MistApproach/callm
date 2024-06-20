type DType = candle_core::DType;

#[derive(Clone, Debug)]
pub enum Device {
    CPU,
    Cuda(usize),
    Metal(usize),
}

/// Describes device ocnfiguration
pub struct DeviceConfig {
    device: Device,
    candle_device: candle_core::Device,
    candle_dtype: DType,
}

impl DeviceConfig {
    pub fn autodetect() -> Self {
        if candle_core::utils::cuda_is_available() {
            Self::new(Device::Cuda(0))
        } else if candle_core::utils::metal_is_available() {
            Self::new(Device::Metal(0))
        } else {
            Self::new(Device::CPU)
        }
    }

    pub fn new(device: Device) -> Self {
        Self {
            candle_dtype: match device {
                Device::CPU => DType::F32,
                Device::Cuda(_) => DType::BF16,
                Device::Metal(_) => DType::F32,
            },
            candle_device: match device {
                Device::CPU => candle_core::Device::Cpu,
                Device::Cuda(n) => {
                    candle_core::Device::new_cuda(n).expect("CUDA device creation error")
                }
                Device::Metal(n) => {
                    candle_core::Device::new_metal(n).expect("Metal device creation error")
                }
            },
            device,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn candle_device(&self) -> &candle_core::Device {
        &self.candle_device
    }

    pub fn candle_dtype(&self) -> DType {
        self.candle_dtype
    }
}

impl std::fmt::Debug for DeviceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self.device)?;
        Ok(())
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::autodetect()
    }
}

impl Clone for DeviceConfig {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            candle_device: self.candle_device.clone(),
            candle_dtype: self.candle_dtype,
        }
    }
}
