#if LUTH_RASTER_ONLY
public uint EvaluateVariant(uint index) { return index + 1; }
#else
[[vk::binding(1, 0)]] RaytracingAccelerationStructure variantScene;
public uint EvaluateVariant(uint index)
{
    RayDesc ray;
    ray.Origin = float3(index, 0, 0);
    ray.Direction = float3(0, 1, 0);
    ray.TMin = 0.01;
    ray.TMax = 100;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(variantScene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed()) {}
    return query.CommittedStatus();
}
#endif
