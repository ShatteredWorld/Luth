// Representative graph consumer: custom decode, shared production opaque shading.
import globals;
import material;
import material_bindings_raster;
import pbr_shade;

MaterialInputs EvalGraph<T : ITexFetch>(GPUMaterialData m, float2 uv0, float2 uv1, T fetch)
{
    MaterialInputs mi = EvalMaterialChannels<T>(m, uv0, uv1, fetch);
    mi.baseColor.rgb *= float3(0.7, 0.8, 0.9);
    return mi;
}
[shader("fragment")]
FOut main(VIn i, bool frontFacing : SV_IsFrontFace, float4 fragCoord : SV_Position)
{
    GPUMaterialData m = GetMaterial(i.materialIndex);
    RasterFetch rf;
    MaterialInputs mi = EvalGraph<RasterFetch>(m, i.uv0, i.uv1, rf);
    return PbrShadeSurface(mi, m, i, frontFacing, fragCoord);
}
