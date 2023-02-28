# quick notes

These are files used with CMake Tools and clangd in `vscode`

`cmake-variants.yaml` should be placed in the top level directory to control build options

`.vscode` contains settings that should be placed in top level directory to build in `build/<CMAKE_BUILD_TYPE>`

`cmake-tools-kits.json` should be updated to point to your needed goma libraries and placed in `$HOME/.local/share/CMakeTools`

As setup the cmake-variants will create `compile_commands.json` when a RelWithDebInfo build is created.

Link that file to your top level goma directory so clangd can see the compile_commands:

    ln -s ./build/RelWithDebInfo/compile_commands.json