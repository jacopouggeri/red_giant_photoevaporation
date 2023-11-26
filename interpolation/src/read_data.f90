module read_data
    use hdf5
    implicit none

    public :: read_hdf5_data

    contains

    subroutine read_hdf5_data(filename, mass, radius, mass_loss, status)
        character(len=*), intent(in) :: filename
        real(kind=8), dimension(:), allocatable, intent(out) :: mass, radius, mass_loss
        integer, intent(out) :: status

        integer(hid_t) :: file_id, dataset_id, dataspace_id
        integer :: i
        integer(hsize_t), dimension(1) :: dims

        ! Initialize HDF5 library
        call h5open_f(status)

        ! Open the file
        call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, status)

        ! Allocate arrays
        dims = [240]  ! Number of elements in each dataset
        allocate(mass(dims(1)))
        allocate(radius(dims(1)))
        allocate(mass_loss(dims(1)))

        ! Read 'mass' dataset
        call h5dopen_f(file_id, 'mass', dataset_id, status)
        call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, mass, dims, status)
        call h5dclose_f(dataset_id, status)

        ! Read 'radius' dataset
        call h5dopen_f(file_id, 'radius', dataset_id, status)
        call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, radius, dims, status)
        call h5dclose_f(dataset_id, status)

        ! Read 'mass_loss' dataset
        call h5dopen_f(file_id, 'mass_loss', dataset_id, status)
        call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, mass_loss, dims, status)
        call h5dclose_f(dataset_id, status)

        ! Clean up
        call h5fclose_f(file_id, status)
        call h5close_f(status)

    end subroutine read_hdf5_data

end module read_data
