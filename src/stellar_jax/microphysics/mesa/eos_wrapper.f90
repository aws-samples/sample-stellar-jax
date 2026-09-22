! Thin Fortran wrapper around MESA's eosDT_get that bypasses gfort2py entirely.
!
! Takes plain real(dp) inputs via bind(C), internally manages MESA's EOS state,
! and returns plain real(dp) outputs. Self-initializes MESA runtime (const, math,
! chem, eos) on first call.
!
! Build: gfortran -shared -fPIC -o libeos_wrapper.so eos_wrapper.f90 \
!        -I$MESA_DIR/include -L$MESA_DIR/lib -leos -lchem -lconst \
!        -lmath -lutils -lnum -lauto_diff
!
! Reference: MESA eos/public/eos_lib.f90 — eosDT_get signature.
module eos_wrapper_mod
   use iso_c_binding, only: c_double, c_int, c_char, c_null_char
   implicit none
   private
   public :: eos_get_plain, eos_wrapper_init

   logical, save :: initialized = .false.
   integer, save :: eos_handle = -1

contains

   subroutine eos_wrapper_init(mesa_dir, mesa_dir_len, ierr) bind(C, name="eos_wrapper_init")
      ! Explicitly initialize the EOS wrapper. Called once from Python.
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

      ! Init order: const → math → chem → eos
      call const_init(trim(mesa_dir_str), ierr)
      if (ierr /= 0) return

      call math_init()

      call chem_init('isotopes.data', ierr)
      if (ierr /= 0) return

      call eos_init(' ', .false., ierr)
      if (ierr /= 0) return

      eos_handle = alloc_eos_handle(ierr)
      if (ierr /= 0) return

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

      subroutine eos_init(cache_dir, use_cache, ierr_out)
         use eos_lib, only: eos_init_ => eos_init
         character(len=*), intent(in) :: cache_dir
         logical, intent(in) :: use_cache
         integer, intent(out) :: ierr_out
         call eos_init_(cache_dir, use_cache, ierr_out)
      end subroutine

      integer function alloc_eos_handle(ierr_out)
         use eos_lib, only: alloc_eos_handle_ => alloc_eos_handle
         integer, intent(out) :: ierr_out
         alloc_eos_handle = alloc_eos_handle_(ierr_out)
      end function
   end subroutine eos_wrapper_init


   subroutine eos_get_plain( &
         logT_in, logRho_in, X_in, Z_in, &
         rho_out, mu_out, nabla_ad_out, S_out, cp_out, &
         chi_rho_out, chi_T_out, lnPgas_out, ierr) bind(C, name="eos_get_plain")
      ! Call MESA eosdt_get for a simple H+He composition.
      !
      ! Inputs: logT, logRho, X (hydrogen mass fraction), Z (metal mass fraction)
      ! Outputs: rho, mu, nabla_ad, entropy, cp, chi_rho, chi_T, lnPgas
      ! ierr: 0 on success, nonzero on failure
      use const_def, only: dp
      use chem_def, only: ih1, ihe4, num_chem_isos
      use eos_lib, only: eosDT_get
      use eos_def, only: num_eos_basic_results, num_eos_d_dxa_results, &
         i_lnPgas, i_mu, i_grad_ad, i_lnS, i_Cp, i_chiRho, i_chiT

      real(c_double), intent(in), value :: logT_in, logRho_in, X_in, Z_in
      real(c_double), intent(out) :: rho_out, mu_out, nabla_ad_out, S_out
      real(c_double), intent(out) :: cp_out, chi_rho_out, chi_T_out
      real(c_double), intent(out) :: lnPgas_out
      integer(c_int), intent(out) :: ierr

      ! Locals
      integer, parameter :: species = 2
      integer, pointer :: chem_id(:), net_iso(:)
      integer, target :: chem_id_ary(2), net_iso_ary(num_chem_isos)
      real(dp) :: xa(species), T, Rho, logT, logRho
      real(dp) :: res(num_eos_basic_results)
      real(dp) :: d_dlnd(num_eos_basic_results)
      real(dp) :: d_dlnT(num_eos_basic_results)
      real(dp) :: d_dxa(num_eos_d_dxa_results, species)
      real(dp) :: Y

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
      T = 10.0_dp ** logT
      Rho = 10.0_dp ** logRho

      ! Call MESA EOS
      call eosDT_get(eos_handle, species, chem_id, net_iso, xa, &
                     Rho, logRho, T, logT, &
                     res, d_dlnd, d_dlnT, d_dxa, ierr)
      if (ierr /= 0) return

      ! Extract results (MESA indices are 1-based)
      rho_out = Rho
      mu_out = res(i_mu)
      nabla_ad_out = res(i_grad_ad)
      S_out = exp(res(i_lnS))
      cp_out = res(i_Cp)
      chi_rho_out = res(i_chiRho)
      chi_T_out = res(i_chiT)
      lnPgas_out = res(i_lnPgas)

   end subroutine eos_get_plain

end module eos_wrapper_mod
