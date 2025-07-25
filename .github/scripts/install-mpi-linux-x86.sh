install_mpi_linux_x86() {
    echo "install mpi for x86"
    dnf install -y mpich mpich-devel openmpi openmpi-devel
}