//! This module provides computation device configuration.

use candle_core::{DType, Device as CandleDevice};

/// Enum representing different types of devices available for computation.
#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    /// The CPU device.
    CPU,
    /// A CUDA device with a specific index.
    Cuda(usize),
    /// A Metal device with a specific index.
    Metal(usize),
}

/// Struct describing the configuration of a device.
#[derive(Clone, Debug)]
pub struct DeviceConfig {
    /// The device type.
    device: Device,
    /// The corresponding Candle device.
    candle_device: CandleDevice,
    /// The data type used by the Candle device.
    candle_dtype: DType,
}

impl DeviceConfig {
    /// Automatically detects the available device and initializes the configuration.
    ///
    /// This function checks for the availability of CUDA and Metal devices. If none are available,
    /// it defaults to using the CPU.
    pub fn autodetect() -> Self {
        if candle_core::utils::cuda_is_available() {
            Self::new(Device::Cuda(0))
        } else if candle_core::utils::metal_is_available() {
            Self::new(Device::Metal(0))
        } else {
            Self::new(Device::CPU)
        }
    }

    /// Creates a new `DeviceConfig` with the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The device type to be used.
    pub fn new(device: Device) -> Self {
        let (candle_device, candle_dtype) = match device {
            Device::CPU => (CandleDevice::Cpu, DType::F32),
            Device::Cuda(n) => (
                CandleDevice::new_cuda(n).expect("CUDA device creation error"),
                DType::BF16,
            ),
            Device::Metal(n) => (
                CandleDevice::new_metal(n).expect("Metal device creation error"),
                DType::F32,
            ),
        };

        Self {
            device,
            candle_device,
            candle_dtype,
        }
    }

    /// Returns a reference to the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the candle device.
    pub fn candle_device(&self) -> &CandleDevice {
        &self.candle_device
    }

    /// Returns the candle data type.
    pub fn candle_dtype(&self) -> DType {
        self.candle_dtype
    }
}

impl Default for DeviceConfig {
    /// Provides a default implementation for `DeviceConfig` by calling `autodetect`.
    fn default() -> Self {
        Self::autodetect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodetect() {
        let config = DeviceConfig::autodetect();
        match config.device() {
            Device::CPU => assert!(config.candle_device().is_cpu()),
            Device::Cuda(_) => assert!(config.candle_device().is_cuda()),
            Device::Metal(_) => assert!(config.candle_device().is_metal()),
        }
    }

    #[test]
    fn test_new_cpu() {
        let config = DeviceConfig::new(Device::CPU);
        assert_eq!(config.device(), &Device::CPU);
        assert!(config.candle_device().is_cpu());
        assert_eq!(config.candle_dtype(), DType::F32);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_new_cuda() {
        let config = DeviceConfig::new(Device::Cuda(0));
        assert_eq!(config.device(), &Device::Cuda(0));
        assert!(config.candle_device().is_cuda());
        assert_eq!(config.candle_dtype(), DType::BF16);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_new_metal() {
        let config = DeviceConfig::new(Device::Metal(0));
        assert_eq!(config.device(), &Device::Metal(0));
        assert!(config.candle_device().is_metal());
        assert_eq!(config.candle_dtype(), DType::F32);
    }

    #[test]
    fn test_default() {
        let config = DeviceConfig::default();
        match config.device() {
            Device::CPU => assert!(config.candle_device().is_cpu()),
            Device::Cuda(_) => assert!(config.candle_device().is_cuda()),
            Device::Metal(_) => assert!(config.candle_device().is_metal()),
        }
    }
}

