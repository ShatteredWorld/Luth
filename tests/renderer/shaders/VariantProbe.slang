import VariantCommon;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> variantOutput;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    variantOutput[tid.x] = EvaluateVariant(tid.x);
}
