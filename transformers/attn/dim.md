Linear layer: nn.Linear(256, 512)
Weight matrix (W): (out_features, in_features) = (512, 256).Bias vector (b): (out_features) = (512).Input tensor (x): Shape (b, seqlen, 256).b: Batch size.seqlen: Sequence length.256: Model size (the size of each input feature vector), which must match in_features. How the multiplication works Preparation for multiplication: The input tensor x of shape (b, seqlen, 256) is conceptually viewed as a collection of b * seqlen vectors, each of size 256.Batched matrix multiplication: The core operation is performed as a batched matrix multiplication. Each 256-dimensional vector from the input is multiplied by the weight matrix.Input "vectors" shape: (b * seqlen, 256)Weight matrix (transposed) shape: (256, 512)Multiplication: (b * seqlen, 256) @ (256, 512)The inner dimensions (256) match and are summed over.The result is a new intermediate tensor of shape (b * seqlen, 512).Reshaping the output: This intermediate tensor is then reshaped back into the original batch and sequence length structure. The dimensions become (b, seqlen, 512).Adding the bias: The bias vector of size (512) is added to the reshaped tensor. This is handled by broadcasting, so the same bias vector is added to every (seqlen) position for every batch. Summary of dimension labels \(\text{Input:\ }(\underbrace{b}_{\text{batch}},\underbrace{\text{seqlen}}_{\text{sequence}},\underbrace{256}_{\text{in\_features}})\quad \rightarrow \quad \text{Output:\ }(\underbrace{b}_{\text{batch}},\underbrace{\text{seqlen}}_{\text{sequence}},\underbrace{512}_{\text{out\_features}})\)


Torch Linear usage:

```python
lin = Linear(input_features, output_features)
# input need only agree with lin per input_features
out = lin(torch.rand(32, input_features))
# out size is (32 x 512)
out = lin(torch.rand(128, 32, input_features))
# out size is (128 x 32 x 512)
```


