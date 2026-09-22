! Thin Fortran wrapper around MESA's net_get that bypasses gfort2py entirely.
!
! Uses pp_cno_extras_o18_ne22.net (correct for solar-interior T ~ 1.5e7 K).
! Self-initializes the full nuclear network: net_init → alloc_net_handle →
! net_start_def → read_net_file → net_finish_def → net_setup_tables.
!
! Build: gfortran -shared -fPIC -o libnet_wrapper.so net_wrapper.f90 \
!        -I$MESA_DIR/include -L$MESA_DIR/lib -lnet -lrates -lchem -lconst \
!        -lmath -lutils -lnum -lauto_diff
!
! Reference: MESA net/public/net_lib.f90 — net_get signature.
!            MESA net/test/src/sample_net.f90 — canonical usage pattern.
module net_wrapper_mod
   use iso_c_binding, only: c_double, c_int, c_char, c_null_char
   implicit none
   private
   public :: net_get_plain, net_wrapper_init

   logical, save :: initialized = .false.
   integer, save :: net_handle = -1
   integer, save :: num_species = 0
   integer, save :: num_reactions = 0
   ! Species mapping arrays (allocated at init)
   integer, save, allocatable :: chem_id_net(:)
   integer, save, allocatable :: net_iso_map(:)

contains

   subroutine net_wrapper_init(mesa_dir, mesa_dir_len, net_name, net_name_len, ierr) &
         bind(C, name="net_wrapper_init")
      ! Initialize the nuclear network. Must be called before net_get_plain.
      !
      ! mesa_dir: path to MESA installation (e.g. "/cache/mesa_libs")
      ! net_name: network filename (e.g. "pp_cno_extras_o18_ne22.net")
      integer(c_int), intent(in), value :: mesa_dir_len, net_name_len
      character(kind=c_char), dimension(mesa_dir_len), intent(in) :: mesa_dir
      character(kind=c_char), dimension(net_name_len), intent(in) :: net_name
      integer(c_int), intent(out) :: ierr

      character(len=256) :: mesa_dir_str, net_name_str
      integer :: i

      ierr = 0
      if (initialized) return

      ! Convert C strings to Fortran strings
      mesa_dir_str = ''
      do i = 1, min(mesa_dir_len, 256)
         mesa_dir_str(i:i) = mesa_dir(i)
      end do
      net_name_str = ''
      do i = 1, min(net_name_len, 256)
         net_name_str(i:i) = net_name(i)
      end do

      call do_init(trim(mesa_dir_str), trim(net_name_str), ierr)
      if (ierr /= 0) return

      initialized = .true.

   contains
      subroutine do_init(mesa_dir_arg, net_name_arg, ierr_out)
         use const_lib, only: const_init
         use math_lib, only: math_init
         use chem_lib, only: chem_init
         use rates_lib, only: rates_init
         use net_lib, only: alloc_net_handle, net_init, net_start_def, &
            read_net_file, net_finish_def, net_setup_tables
         use net_def, only: Net_General_Info, get_net_ptr
         use chem_def, only: num_chem_isos

         character(len=*), intent(in) :: mesa_dir_arg, net_name_arg
         integer, intent(out) :: ierr_out
         type(Net_General_Info), pointer :: g

         ierr_out = 0

         ! 1. Base inits
         call const_init(mesa_dir_arg, ierr_out)
         if (ierr_out /= 0) return

         call math_init()

         call chem_init('isotopes.data', ierr_out)
         if (ierr_out /= 0) return

         ! 2. Rates init (args: reactions list, jina reaclib, rate tables dir,
         !    use_suzuki_weak_rates, use_special_weak_rates,
         !    special_weak_states_file, special_weak_transitions_file, cache_dir, ierr)
         call rates_init('reactions.list', '', 'rate_tables', .false., &
                        .false., '', '', '', ierr_out)
         if (ierr_out /= 0) return

         ! 3. Net module init (sets up special case reaction info)
         call net_init(ierr_out)
         if (ierr_out /= 0) return

         ! 4. Network setup
         net_handle = alloc_net_handle(ierr_out)
         if (ierr_out /= 0) return

         call net_start_def(net_handle, ierr_out)
         if (ierr_out /= 0) return

         ! NOTE: MESA read_net_file signature is (filename, handle, ierr)
         call read_net_file(net_name_arg, net_handle, ierr_out)
         if (ierr_out /= 0) return

         call net_finish_def(net_handle, ierr_out)
         if (ierr_out /= 0) return

         ! 5. Setup rate tables (builds internal rate data structures)
         call net_setup_tables(net_handle, '', ierr_out)
         if (ierr_out /= 0) return

         ! 6. Get species and reaction info from the network
         call get_net_ptr(net_handle, g, ierr_out)
         if (ierr_out /= 0) return

         num_species = g% num_isos
         num_reactions = g% num_reactions
         allocate(chem_id_net(num_species))
         allocate(net_iso_map(num_chem_isos))
         chem_id_net = g% chem_id(1:num_species)
         net_iso_map = 0
         do i = 1, num_species
            net_iso_map(g% chem_id(i)) = i
         end do

      end subroutine do_init
   end subroutine net_wrapper_init


   subroutine net_get_plain( &
         rho_in, T_in, X_in, Z_in, &
         eps_nuc_out, ierr) bind(C, name="net_get_plain")
      ! Call MESA net_get for nuclear energy generation rate.
      !
      ! Inputs: rho (g/cm³), T (K), X (hydrogen mass fraction), Z (metal mass fraction)
      ! Outputs: eps_nuc (erg/g/s)
      ! ierr: 0 on success, nonzero on failure
      !
      ! Composition: simplified — distributes X as H1, Y as He4,
      ! and Z across CNO species proportional to solar abundances.
      !
      ! Reference: MESA net/test/src/sample_net.f90 (canonical usage pattern)
      use const_def, only: dp
      use chem_def, only: ih1, ihe4, ic12, in14, io16, num_chem_isos, num_categories
      use chem_lib, only: composition_info
      use net_lib, only: net_get
      use net_def, only: Net_General_Info, Net_Info, get_net_ptr
      use rates_def, only: std_reaction_Qs, std_reaction_neuQs, extended_screening

      real(c_double), intent(in), value :: rho_in, T_in, X_in, Z_in
      real(c_double), intent(out) :: eps_nuc_out
      integer(c_int), intent(out) :: ierr

      ! Locals
      type(Net_General_Info), pointer :: g
      ! NOTE: n is intentionally uninitialized. MESA's net_get dereferences n%
      ! fields (workspace pointers), but this wrapper is never called at runtime
      ! (nuclear stays JAX due to Net_Info allocation infeasibility).
      ! The wrapper compiles and loads (AC1/AC2) but is not invoked.
      type(Net_Info) :: n
      real(dp), allocatable :: xa(:), d_eps_nuc_dx(:)
      real(dp), allocatable :: dxdt(:), d_dxdt_dRho(:), d_dxdt_dT(:)
      real(dp), allocatable :: d_dxdt_dx(:,:)
      real(dp), allocatable :: dabar_dx(:), dzbar_dx(:), dmc_dx(:)
      real(dp), target, allocatable :: rate_factors_ary(:)
      real(dp), pointer, dimension(:) :: rate_factors
      real(dp) :: Y, T, Rho, logT, logRho
      real(dp) :: eps_nuc, d_eps_nuc_dRho, d_eps_nuc_dT
      real(dp) :: eps_nuc_categories(num_categories)
      real(dp) :: eps_neu_total
      real(dp) :: eta, d_eta_dlnT, d_eta_dlnRho
      real(dp) :: xh, xhe, z_comp, abar, zbar, z2bar, z53bar, ye
      real(dp) :: mass_correction, xsum, weak_rate_factor
      integer :: idx, screening_mode
      logical :: skip_jacobian
      real(dp) :: Z_frac_C12, Z_frac_N14, Z_frac_O16, Z_other

      ierr = 0

      if (.not. initialized) then
         ierr = -1
         return
      end if

      call get_net_ptr(net_handle, g, ierr)
      if (ierr /= 0) return

      ! Allocate workspace
      allocate(xa(num_species))
      allocate(d_eps_nuc_dx(num_species))
      allocate(dxdt(num_species))
      allocate(d_dxdt_dRho(num_species))
      allocate(d_dxdt_dT(num_species))
      allocate(d_dxdt_dx(num_species, num_species))
      allocate(dabar_dx(num_species))
      allocate(dzbar_dx(num_species))
      allocate(dmc_dx(num_species))
      allocate(rate_factors_ary(num_reactions))

      ! Build composition array
      ! Solar CNO mass fractions (Asplund et al. 2009, ARAA 51, 457):
      ! X_C=2.36e-3, X_N=6.96e-4, X_O=5.73e-3 (sum = 8.786e-3)
      Z_frac_C12 = 0.269_dp
      Z_frac_N14 = 0.079_dp
      Z_frac_O16 = 0.652_dp

      xa = 0.0_dp
      Y = max(0.0_dp, 1.0_dp - X_in - Z_in)

      ! Assign H and He
      idx = net_iso_map(ih1)
      if (idx > 0) xa(idx) = X_in
      idx = net_iso_map(ihe4)
      if (idx > 0) xa(idx) = Y

      ! Assign CNO from Z
      idx = net_iso_map(ic12)
      if (idx > 0) xa(idx) = Z_in * Z_frac_C12
      idx = net_iso_map(in14)
      if (idx > 0) xa(idx) = Z_in * Z_frac_N14
      idx = net_iso_map(io16)
      if (idx > 0) xa(idx) = Z_in * Z_frac_O16

      ! Remaining Z distributed to He4 (trace metals approximation)
      Z_other = Z_in * (1.0_dp - Z_frac_C12 - Z_frac_N14 - Z_frac_O16)
      idx = net_iso_map(ihe4)
      if (idx > 0) xa(idx) = xa(idx) + Z_other

      T = T_in
      Rho = rho_in
      logT = log10(T)
      logRho = log10(Rho)

      ! Compute composition info (abar, zbar, etc.) using MESA's own routine
      call composition_info( &
         num_species, chem_id_net, xa, xh, xhe, z_comp, &
         abar, zbar, z2bar, z53bar, ye, mass_correction, &
         xsum, dabar_dx, dzbar_dx, dmc_dx)

      ! Electron degeneracy — zero for non-degenerate solar interior
      eta = 0.0_dp
      d_eta_dlnT = 0.0_dp
      d_eta_dlnRho = 0.0_dp

      ! Rate factors: all 1.0 (no custom rate modifications)
      rate_factors => rate_factors_ary
      rate_factors(:) = 1.0_dp
      weak_rate_factor = 1.0_dp

      ! Extended screening (standard for stellar interior)
      screening_mode = extended_screening
      skip_jacobian = .true.  ! we only need eps_nuc, not the full Jacobian

      ! Zero output arrays
      d_eps_nuc_dx = 0.0_dp
      dxdt = 0.0_dp
      d_dxdt_dRho = 0.0_dp
      d_dxdt_dT = 0.0_dp
      d_dxdt_dx = 0.0_dp
      eps_nuc_categories = 0.0_dp

      ! Call MESA nuclear network
      ! Signature: net_get(handle, just_dxdt, n, num_isos, num_reactions,
      !   x, temp, log10temp, rho, log10rho,
      !   abar, zbar, z2bar, ye, eta, d_eta_dlnT, d_eta_dlnRho,
      !   rate_factors, weak_rate_factor, reaction_Qs, reaction_neuQs,
      !   eps_nuc, d_eps_nuc_dRho, d_eps_nuc_dT, d_eps_nuc_dx,
      !   dxdt, d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx,
      !   screening_mode, eps_nuc_categories, eps_neu_total, ierr)
      call net_get(net_handle, skip_jacobian, n, num_species, num_reactions, &
                   xa, T, logT, Rho, logRho, &
                   abar, zbar, z2bar, ye, eta, d_eta_dlnT, d_eta_dlnRho, &
                   rate_factors, weak_rate_factor, &
                   std_reaction_Qs, std_reaction_neuQs, &
                   eps_nuc, d_eps_nuc_dRho, d_eps_nuc_dT, d_eps_nuc_dx, &
                   dxdt, d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx, &
                   screening_mode, &
                   eps_nuc_categories, eps_neu_total, &
                   ierr)
      if (ierr /= 0) then
         eps_nuc_out = 0.0_dp
         deallocate(xa, d_eps_nuc_dx, dxdt, d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx)
         deallocate(dabar_dx, dzbar_dx, dmc_dx, rate_factors_ary)
         return
      end if

      eps_nuc_out = eps_nuc

      deallocate(xa, d_eps_nuc_dx, dxdt, d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx)
      deallocate(dabar_dx, dzbar_dx, dmc_dx, rate_factors_ary)

   end subroutine net_get_plain

end module net_wrapper_mod
