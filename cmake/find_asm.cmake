include_guard(GLOBAL)

if(WIN32)
    find_program(LUTH_MASM_COMPILER NAMES ml64 ml)
    if(NOT LUTH_MASM_COMPILER)
        find_program(LUTH_VSWHERE
            NAMES vswhere
            HINTS "$ENV{SystemDrive}/Program Files (x86)/Microsoft Visual Studio/Installer"
        )
        if(LUTH_VSWHERE)
            execute_process(
                COMMAND "${LUTH_VSWHERE}" -latest -products *
                    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
                    -property installationPath
                OUTPUT_VARIABLE LUTH_VS_INSTALLATION
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            file(GLOB LUTH_MASM_CANDIDATES
                "${LUTH_VS_INSTALLATION}/VC/Tools/MSVC/*/bin/Hostx64/x64/ml64.exe"
                "${LUTH_VS_INSTALLATION}/VC/Tools/MSVC/*/bin/Hostx86/x64/ml64.exe"
            )
            if (LUTH_MASM_CANDIDATES)
                list(SORT LUTH_MASM_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
                list(GET LUTH_MASM_CANDIDATES 0 LUTH_MASM_COMPILER)
            endif()
        endif()
    endif()
    if(NOT LUTH_MASM_COMPILER)
        message(FATAL_ERROR
            "The Visual Studio x64 MASM assembler (ml64.exe) is required for Windows builds"
        )
    endif()
    set(LUTH_MASM_COMPILER "${LUTH_MASM_COMPILER}" CACHE FILEPATH "x64 MASM compiler" FORCE)
    set(CMAKE_ASM_MASM_COMPILER "${LUTH_MASM_COMPILER}" CACHE FILEPATH "MASM compiler" FORCE)
    enable_language(ASM_MASM)
endif()