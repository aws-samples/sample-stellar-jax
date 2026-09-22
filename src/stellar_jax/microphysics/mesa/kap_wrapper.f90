! Thin Fortran wrapper around MESA's kap_get that bypasses gfort2py entirely.
!
! Takes plain real(dp) inputs via bind(C), internally manages MESA's kap state,
! and returns plain real(dp) outputs. Self-initializes MESA runtime (const, math,
! chem, kap) on first call.
!
! Build: gfortran -shared -fPIC -o libkap_wrapper.so kap_wrapper.f90 \
!        -I$MESA_DIR/include -L$MESA_DIR/lib -lkap -lchem -lconst \
!        -lmath -lutils -lnum -lauto_diff
!
! Reference: MESA kap/public/kap_lib.f90 — kap_get signature.
module kap_wrapper_mod
   use iso_c_binding, only: c_double, c_int, c_char, c_null_char
   implicit none
   private
   public :: kap_get_plain, kap_wrapper_init

   logical, save :: initialized = .false.
   integer, save :: kap_handle = -1

contains

   subroutine kap_wrapper_init(mesa_dir, mesa_dir_len, Zbase, ierr) bind(C, name="kap_wrapper_init")
      ! Explicitly initialize the kap wrapper. Called once from Python.
      ! Zbase must be set on the handle for MESA opacity to work.
      integer(c_int), intent(in), value :: mesa_dir_len
      character(kind=c_char), dimension(mesa_dir_len), intent(in) :: mesa_dir
      real(c_double), intent(in), value :: Zbase
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

      ! Init order: const → math → chem → kap
      call const_init(trim(mesa_dir_str), ierr)
      if (ierr /= 0) return

      call math_init()

      call chem_init('isotopes.data', ierr)
      if (ierr /= 0) return

      call kap_init_sub(.false., ' ', ierr)
      if (ierr /= 0) return

      kap_handle = alloc_kap_handle_sub(ierr)
      if (ierr /= 0) return

      ! Set Zbase on the handle (required by kap_get)
      call set_kap_zbase(kap_handle, Zbase)

      initialized = .true.

   contains
      subroutine const_init(mesa_dir_arg, ierr_out)
         use const_lib, only: const_init_ => const_init
         character(len=*), intent(in) :: mesa_dir_arg
         integer, intent(out) :: ierr_out
         call const_init_(mesa_dir_arg, ierr_out)
      end subroutine

      subroutine math_init()
         use math_lib, only: math_init_ => math_init
         call math_init_()
      end subroutine

      subroutine chem_init(isotopes_file, ierr_out)
         use chem_lib, only: chem_init_ => chem_init
         character(len=*), intent(in) :: isotopes_file
         integer, intent(out) :: ierr_out
         call chem_init_(isotopes_file, ierr_out)
      end subroutine

      subroutine kap_init_sub(use_cache, cache_dir, ierr_out)
         use kap_lib, only: kap_init
         logical, intent(in) :: use_cache
         character(len=*), intent(in) :: cache_dir
         integer, intent(out) :: ierr_out
         call kap_init(use_cache, cache_dir, ierr_out)
      end subroutine

      integer function alloc_kap_handle_sub(ierr_out)
         use kap_lib, only: alloc_kap_handle_ => alloc_kap_handle
         integer, intent(out) :: ierr_out
         alloc_kap_handle_sub = alloc_kap_handle_(ierr_out)
      end function

      subroutine set_kap_zbase(handle, zbase_val)
         use kap_lib, only: kap_ptr
         use kap_def, only: Kap_General_Info
         integer, intent(in) :: handle
         real(c_double), intent(in) :: zbase_val
         type(Kap_General_Info), pointer :: kap_info
         integer :: ierr_local
         call kap_ptr(handle, kap_info, ierr_local)
         if (ierr_local == 0) kap_info% Zbase = zbase_val
      end subroutine
   end subroutine kap_wrapper_init


   subroutine kap_get_plain( &
         logT_in, logRho_in, X_in, Z_in, &
         log_kappa_out, dlnkap_dlnT_out, dlnkap_dlnRho_out, &
         ierr) bind(C, name="kap_get_plain")
      ! Call MESA kap_get for a simple H+He composition.
      !
      ! Inputs: logT, logRho, X (hydrogen mass fraction), Z (metal mass fraction)
      ! Outputs: log10(kappa), d(ln kappa)/d(ln T), d(ln kappa)/d(ln rho)
      ! ierr: 0 on success, nonzero on failure
      use const_def, only: dp
      use chem_def, only: ih1, ihe4, num_chem_isos
      use kap_def, only: num_kap_fracs
      use kap_lib, only: kap_get

      real(c_double), intent(in), value :: logT_in, logRho_in, X_in, Z_in
      real(c_double), intent(out) :: log_kappa_out, dlnkap_dlnT_out, dlnkap_dlnRho_out
      integer(c_int), intent(out) :: ierr

      ! Locals
      integer, parameter :: species = 2
      integer, pointer :: chem_id(:), net_iso(:)
      integer, target :: chem_id_ary(2), net_iso_ary(num_chem_isos)
      real(dp) :: xa(species)
      real(dp) :: logT, logRho, Y
      real(dp) :: kap_out, dlnkap_dlnRho, dlnkap_dlnT
      real(dp) :: kap_fracs(num_kap_fracs), dlnkap_dxa(species)
      real(dp) :: lnfree_e, d_lnfree_e_dlnRho, d_lnfree_e_dlnT
      real(dp) :: eta, d_eta_dlnRho, d_eta_dlnT

      ierr = 0

      if (.not. initialized) then
         ierr = -1
         return
      end if

      ! Set up composition — MESA requires pointer args for chem_id/net_iso
      chem_id_ary(1) = ih1
      chem_id_ary(2) = ihe4
      chem_id => chem_id_ary

      net_iso_ary = 0
      net_iso_ary(ih1) = 1
      net_iso_ary(ihe4) = 2
      net_iso => net_iso_ary

      Y = max(0.0_dp, 1.0_dp - X_in - Z_in)
      xa(1) = X_in
      xa(2) = Y

      logT = logT_in
      logRho = logRho_in

      ! Electron parameters (zero for simple composition — MESA will compute internally)
      lnfree_e = 0.0_dp
      d_lnfree_e_dlnRho = 0.0_dp
      d_lnfree_e_dlnT = 0.0_dp
      eta = 0.0_dp
      d_eta_dlnRho = 0.0_dp
      d_eta_dlnT = 0.0_dp

      ! Call MESA opacity
      call kap_get(kap_handle, species, chem_id, net_iso, xa, &
                   logRho, logT, &
                   lnfree_e, d_lnfree_e_dlnRho, d_lnfree_e_dlnT, &
                   eta, d_eta_dlnRho, d_eta_dlnT, &
                   kap_fracs, kap_out, dlnkap_dlnRho, dlnkap_dlnT, &
                   dlnkap_dxa, ierr)
      if (ierr /= 0) return

      ! kap_out is LINEAR opacity (cm²/g) from MESA — convert to log10
      log_kappa_out = log10(kap_out)
      dlnkap_dlnT_out = dlnkap_dlnT
      dlnkap_dlnRho_out = dlnkap_dlnRho

   end subroutine kap_get_plain

end module kap_wrapper_mod
