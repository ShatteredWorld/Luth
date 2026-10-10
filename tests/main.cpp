// doctest entry for LuthTests.
// doctest single-header vendored at tests/extern/doctest/doctest.h
// (commit history records the pinned upstream tag + download URL).

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "luth/core/diagnostics/Log.h"
#include "luth/core/types/LuthMath.h"
#include "luth/assets/FileSystem.h"

int main(int argc, char** argv)
{
    Luth::Log::Init();
    // Production shader tests resolve the same common/registry roots as the engine.
    Luth::FileSystem::InitEngine(std::filesystem::path(__FILE__).parent_path().parent_path() / "engine");

    doctest::Context ctx(argc, argv);
    return ctx.run();
}
