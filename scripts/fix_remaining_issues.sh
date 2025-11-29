#!/bin/bash
# Fix remaining cudarc migration issues

cd /mnt/c/Users/Predator/Desktop/PRISM/crates/prism-gpu/src

# Fix CudaDevice in documentation examples
find . -name "*.rs" -exec sed -i 's/CudaDevice::new(/CudaContext::new(/g' {} \;
find . -name "*.rs" -exec sed -i 's/Arc<CudaDevice>/Arc<CudaContext>/g' {} \;
find . -name "*.rs" -exec sed -i 's/: Arc<CudaDevice>/: Arc<CudaContext>/g' {} \;

# Fix variable names in examples and tests
find . -name "*.rs" -exec sed -i 's/let device = CudaContext/let context = CudaContext/g' {} \;
find . -name "*.rs" -exec sed -i 's/Arc::new(device)/Arc::new(context)/g' {} \;

# Fix function parameters
find . -name "*.rs" -exec sed -i 's/pub fn new(device: Arc<CudaContext>/pub fn new(context: Arc<CudaContext>/g' {} \;
find . -name "*.rs" -exec sed -i 's/device: Arc<CudaContext>/context: Arc<CudaContext>/g' {} \;

# Fix struct field references in builder patterns
find . -name "*.rs" -exec sed -i 's/device: Arc<CudaDevice>/context: Arc<CudaContext>/g' {} \;

# Fix primary_device methods to return CudaContext
find . -name "*.rs" -exec sed -i 's/pub fn primary_device(&self) -> Arc<CudaDevice>/pub fn primary_device(&self) -> Arc<CudaContext>/g' {} \;

echo "Fixed remaining CudaDevice references"
