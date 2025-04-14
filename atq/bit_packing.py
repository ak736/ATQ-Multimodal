import torch
import numpy as np

class TernaryBitPacking:
    """
    Utility class for packing ternary weights into a memory-efficient bitpacked format.
    
    This class provides methods to pack ternary weights (-1, 0, 1) into a compact 
    bit representation, where each ternary value is encoded using just 2 bits:
    
    * -1 is encoded as 00
    *  0 is encoded as 01
    * +1 is encoded as 10
    
    This allows storing 4 ternary values in a single byte, resulting in a theoretical
    16x compression compared to standard float32 representation.
    
    For deployment on edge devices, this provides significant memory savings.
    """
    
    @staticmethod
    def pack_ternary_weights(ternary_weights):
        """
        Pack ternary weights into a bit-packed representation.
        
        Args:
            ternary_weights: Tensor of ternary weights (-1, 0, 1)
            
        Returns:
            Dictionary containing:
                - packed_weights: Bit-packed tensor (uint8)
                - original_shape: Shape of original tensor for unpacking
                - metadata: Additional info for reconstruction
        """
        # Validate input contains only ternary values
        unique_values = torch.unique(ternary_weights)
        valid_values = torch.tensor([-1.0, 0.0, 1.0])
        if not all(v in valid_values for v in unique_values):
            raise ValueError("Input must contain only ternary values (-1, 0, 1)")
        
        # Store original shape for unpacking
        original_shape = ternary_weights.shape
        
        # Convert to contiguous flat array for easier processing
        flat_weights = ternary_weights.reshape(-1).contiguous()
        
        # Map ternary values to 2-bit representation: -1 → 0, 0 → 1, 1 → 2
        # We add 1 to shift from [-1,0,1] to [0,1,2]
        mapped_values = (flat_weights + 1).to(torch.uint8)
        
        # Calculate packed tensor dimensions
        num_values = flat_weights.numel()
        packed_size = (num_values + 3) // 4  # Each byte holds 4 ternary values
        
        # Create packed tensor initialized with zeros
        packed_weights = torch.zeros(packed_size, dtype=torch.uint8, 
                                    device=ternary_weights.device)
        
        # Pack values into bytes - each byte contains 4 ternary values
        for i in range(min(num_values, packed_size * 4)):
            # Get bit position within the byte
            byte_idx = i // 4
            bit_pos = (i % 4) * 2
            
            # Get value to pack (0, 1, or 2)
            val = mapped_values[i].item()
            
            # Insert at correct bit position
            packed_weights[byte_idx] |= (val << bit_pos)
        
        # Store metadata for unpacking
        metadata = {
            'num_values': num_values,
            'encoding': {
                0: -1,  # 00 → -1
                1: 0,   # 01 → 0
                2: 1    # 10 → +1
            }
        }
        
        return {
            'packed_weights': packed_weights,
            'original_shape': original_shape,
            'metadata': metadata
        }
    
    @staticmethod
    def unpack_ternary_weights(packed_data):
        """
        Unpack weights from bit-packed representation back to ternary values.
        
        Args:
            packed_data: Dictionary containing packed weights and metadata
            
        Returns:
            Tensor of ternary weights (-1, 0, 1)
        """
        packed_weights = packed_data['packed_weights']
        original_shape = packed_data['original_shape']
        metadata = packed_data['metadata']
        num_values = metadata['num_values']
        
        # Create output tensor
        unpacked = torch.zeros(num_values, dtype=torch.float,
                              device=packed_weights.device)
        
        # Unpack values
        for i in range(num_values):
            byte_idx = i // 4
            bit_pos = (i % 4) * 2
            
            # Extract 2-bit value (0, 1, or 2)
            val = (packed_weights[byte_idx] >> bit_pos) & 0x3
            
            # Map back to ternary (-1, 0, 1)
            unpacked[i] = metadata['encoding'][val.item()]
        
        # Reshape to original dimensions
        return unpacked.reshape(original_shape)

    @staticmethod
    def compute_memory_savings(original_tensor):
        """
        Calculate theoretical memory savings from bit-packing.
        
        Args:
            original_tensor: Original float tensor
            
        Returns:
            Dictionary with memory usage statistics
        """
        # Calculate original memory usage (float32)
        original_bytes = original_tensor.numel() * 4
        
        # Calculate packed memory usage (2 bits per value)
        packed_bytes = (original_tensor.numel() * 2 + 7) // 8
        
        # Calculate compression ratio
        compression_ratio = original_bytes / packed_bytes
        
        return {
            'original_bytes': original_bytes,
            'packed_bytes': packed_bytes,
            'compression_ratio': compression_ratio,
            'memory_reduction': 1.0 - (packed_bytes / original_bytes)
        }

    @staticmethod
    def fast_ternary_matmul(packed_data, input_tensor, alpha=1.0):
        """
        Simulated fast matrix multiplication with packed ternary weights.
        
        NOTE: This is a reference implementation for demonstration.
        For actual speed gains, this would need native implementation.
        
        Args:
            packed_data: Packed weights
            input_tensor: Input tensor for matrix multiplication
            alpha: Scaling factor for weights
            
        Returns:
            Result of matrix multiplication
        """
        # Unpack weights (in a real implementation, you'd operate directly on packed format)
        weights = TernaryBitPacking.unpack_ternary_weights(packed_data)
        
        # Reshape for matmul
        if len(weights.shape) == 2 and len(input_tensor.shape) == 2:
            # Standard matmul: [batch, in] @ [out, in]T = [batch, out]
            result = input_tensor @ weights.t()
        else:
            # Handle other cases as needed
            result = torch.matmul(input_tensor, weights.t())
        
        # Apply scaling factor
        return result * alpha


# Example usage:
"""
# Create a tensor of ternary weights
ternary_weights = torch.tensor([
    [-1, 0, 1, -1],
    [0, 1, -1, 0],
    [1, -1, 0, 1]
], dtype=torch.float)

# Pack the weights
packed_data = TernaryBitPacking.pack_ternary_weights(ternary_weights)

# Check memory savings
savings = TernaryBitPacking.compute_memory_savings(ternary_weights)
print(f"Compression ratio: {savings['compression_ratio']:.2f}x")
print(f"Memory reduction: {savings['memory_reduction']*100:.1f}%")

# Unpack to verify correctness
unpacked = TernaryBitPacking.unpack_ternary_weights(packed_data)
print(f"Original:\n{ternary_weights}")
print(f"Unpacked:\n{unpacked}")
print(f"Identical: {torch.all(ternary_weights == unpacked).item()}")

# Demonstrate matrix multiplication
input_tensor = torch.randn(2, 4)  # [batch_size, in_features]
result = TernaryBitPacking.fast_ternary_matmul(packed_data, input_tensor, alpha=2.0)
print(f"Input shape: {input_tensor.shape}")
print(f"Result shape: {result.shape}")
"""