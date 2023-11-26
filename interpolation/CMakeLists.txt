cmake_minimum_required(VERSION 3.5)
project(mass_loss_interpolator LANGUAGES Fortran)

# Find a Fortran compiler (e.g., gfortran)
enable_language(Fortran)

# Add your Fortran source files
add_subdirectory(src)