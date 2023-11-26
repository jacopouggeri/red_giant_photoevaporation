program interp
    use read_data
    implicit none

    character(len=*), parameter :: filename = 'data/mass_loss_data_m_dot_0.32.h5'
    real(kind=8), dimension(:), allocatable :: mass, radius, mass_loss
    integer :: status
    integer :: i

    ! Call the HDF5 data reading function
    call read_hdf5_data(filename, mass, radius, mass_loss, status)

    ! Print a sample of the data (optional)
    print *, 'Sample data:'
    do i = 1, min(10, size(mass))
        print *, 'Mass:', mass(i), ' Radius:', radius(i), ' Mass Loss:', mass_loss(i)
    end do

    ! Clean up (optional)
    deallocate(mass, radius, mass_loss)

end program interp
