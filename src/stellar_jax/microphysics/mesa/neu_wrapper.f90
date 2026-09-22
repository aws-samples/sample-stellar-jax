! Thin Fortran wrapper around MESA's neu_get that bypasses gfort2py entirely.
!
! Takes plain real(dp) inputs via bind(C), calls MESA's neutrino loss routines,
! and returns the total neutrino loss rate (erg/g/s). Self-initializes MESA
! runtime (const, math, chem) on first call.
!
! Build: gfortran -shared -fPIC -o libneu_wrapper.so neu_wrapper.f90 \
!        -I$MESA_DIR/include -L$MESA_DIR/lib -lneu -lchem -lconst \
!        -lmath -lutils -lnum -lauto_diff
!
! Reference: MESA neu/public/neu_lib.f90 — neu_get signature:
!   subroutine neu_get(T, log10_T, Rho, log10_Rho, abar, zbar,
!                      log10_Tlim, flags, loss, sources, info)
module neu_wrapper_mod
   use iso_c_binding, only: c_double, c_int, c_char, c_null_char
   implicit none
   private
   public :: neu_get_plain, neu_wrapper_init

   logical, save :: initialized = .false.

contains

   subroutine neu_wrapper_init(mesa_dir, mesa_dir_len, ierr) bind(C, name="neu_wrapper_init")
      ! Explicitly initialize the neutrino wrapper. Called once from Python.
      integer(c_int), intent(in), value :: mesa_dir_len
      character(kind=c_char), dimension(mesa_dir_len), intent(in) :: mesa_dir
      integer(c_int), intent(out) :: ierr

      character(len=256) :: mesa_dir_str
      integer :: i

      ierr = 0
      if (initialized) return

      ! Convert C char array to Fortran string
      mesa_dir_str = ''
      do i = 1, min(mesa_dir_len, 256)
         mesa_dir_str(i:i) = mesa_dir(i)
      end do

      ! Init order: const → math → chem (neu uses chem for composition info)
      call do_const_init(trim(mesa_dir_str), ierr)
      if (ierr /= 0) return

      call do_math_init()

      call do_chem_init('isotopes.data', ierr)
      if (ierr /= 0) return

      initialized = .true.

   contains
      subroutine do_const_init(mesa_dir_arg, ierr_out)
         use const_lib, only: const_init
         character(len=*), intent(in) :: mesa_dir_arg
         integer, intent(out) :: ierr_out
         call const_init(mesa_dir_arg, ierr_out)
      end subroutine

      subroutine do_math_init()
         use math_lib, only: math_init
         call math_init()
      end subroutine

      subroutine do_chem_init(isotopes_file, ierr_out)
         use chem_lib, only: chem_init
         character(len=*), intent(in) :: isotopes_file
         integer, intent(out) :: ierr_out
         call chem_init(isotopes_file, ierr_out)
      end subroutine
   end subroutine neu_wrapper_init


   subroutine neu_get_plain( &
         logT_in, logRho_in, X_in, Z_in, &
         eps_neu_out, ierr) bind(C, name="neu_get_plain")
      ! Call MESA neu_get for neutrino energy loss rate.
      !
      ! Inputs: logT, logRho, X (hydrogen mass fraction), Z (metal mass fraction)
      ! Outputs: eps_neu (total neutrino loss in erg/g/s, POSITIVE = energy lost)
      ! ierr: 0 on success, nonzero on failure
      !
      ! Reference: MESA neu/public/neu_lib.f90
      !   neu_get(T, log10_T, Rho, log10_Rho, abar, zbar,
      !           log10_Tlim, flags, loss, sources, info)
      use const_def, only: dp
      use neu_lib, only: neu_get
      use neu_def, only: num_neu_types, num_neu_rvs

      real(c_double), intent(in), value :: logT_in, logRho_in, X_in, Z_in
      real(c_double), intent(out) :: eps_neu_out
      integer(c_int), intent(out) :: ierr

      ! Locals
      real(dp) :: T, logT, Rho, logRho, Y
      real(dp) :: abar, zbar, log10_Tlim
      real(dp) :: loss(num_neu_rvs)
      real(dp) :: sources(num_neu_types, num_neu_rvs)
      logical :: flags(num_neu_types)
      integer :: info

      ierr = 0

      if (.not. initialized) then
         ierr = -1
         return
      end if

      logT = logT_in
      logRho = logRho_in
      T = 10.0_dp ** logT
      Rho = 10.0_dp ** logRho

      ! Compute abar, zbar from simple H+He composition
      ! abar = 1 / sum(X_i / A_i)  — mean atomic mass number
      ! zbar = sum(X_i * Z_i / A_i) / sum(X_i / A_i)
      ! For H (A=1, Z=1) + He (A=4, Z=2):
      Y = max(0.0_dp, 1.0_dp - X_in - Z_in)
      ! sum(X_i / A_i) = X/1 + Y/4 + Z/~16 (assume metals ~ O16)
      ! For simplicity with the two-species approximation:
      abar = 1.0_dp / (X_in / 1.0_dp + Y / 4.0_dp + Z_in / 16.0_dp)
      zbar = (X_in * 1.0_dp + Y * 2.0_dp / 4.0_dp + Z_in * 8.0_dp / 16.0_dp) * abar

      ! log10_Tlim: temperature below which neutrino losses are smoothly tapered to 0
      ! 7.5 is the standard choice (Itoh et al. data valid above 10^7 K)
      log10_Tlim = 7.5_dp

      ! Enable all neutrino processes
      flags = .true.

      ! Zero the output arrays (they are intent(inout))
      loss = 0.0_dp
      sources = 0.0_dp

      ! Call MESA neutrino
      call neu_get(T, logT, Rho, logRho, abar, zbar, &
                   log10_Tlim, flags, loss, sources, info)
      if (info /= 0) then
         ierr = info
         eps_neu_out = 0.0_dp
         return
      end if

      ! loss(1) = total neutrino loss rate (erg/g/s)
      ! In MESA convention this is the loss RATE (positive = energy lost from star)
      eps_neu_out = loss(1)

   end subroutine neu_get_plain

end module neu_wrapper_mod
