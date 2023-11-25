program read_hdf5
    use hdf5
    implicit none

    character(len=*), parameter :: filename = 'data/mass_loss_data_m_dot_0.32.h5'
    integer(hid_t) :: file_id, dataset_id, dataspace_id
    integer :: status
    integer(hsize_t), dimension(1) :: dims
    real(kind=8), dimension(:), allocatable :: mass, radius, mass_loss
    integer :: i

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

    ! Close the file
    call h5fclose_f(file_id, status)

    ! Finalize HDF5 library
    call h5close_f(status)

    ! Print a sample of the data (optional)
    print *, 'Sample data:'
    do i = 1, min(10, size(mass))
        print *, 'Mass:', mass(i), ' Radius:', radius(i), ' Mass Loss:', mass_loss(i)
    end do

    ! Clean up
    deallocate(mass, radius, mass_loss)

end program read_hdf5