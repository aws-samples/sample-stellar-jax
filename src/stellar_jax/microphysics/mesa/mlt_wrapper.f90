! Thin Fortran wrapper around MESA's set_mlt that bypasses gfort2py's inability
! to extract auto_diff_real_star_order1 output fields.
!
! Takes plain real(dp) inputs (the %val part only), constructs the auto_diff types,
! calls set_mlt, and returns %val of each output as plain real(dp).
!
! Build: gfortran -shared -fPIC -o libmlt_wrapper.so mlt_wrapper.f90 \
!        -I$MESA_DIR/include -L$MESA_DIR/lib -lturb -leos -lkap -lchem -lconst \
!        -lauto_diff -lutils -lmath -lnum
!
! Reference: MESA turb/public/turb.f90 — set_MLT signature.
module mlt_wrapper_mod
   use iso_c_binding, only: c_double, c_int, c_char, c_null_char
   implicit none
   private
   public :: set_mlt_plain
contains

   subroutine set_mlt_plain( &
         mlt_option_len, mlt_option_chars, &
         mixing_length_alpha, Henyey_MLT_nu_param, Henyey_MLT_y_param, max_conv_vel, &
         chiT_val, chiRho_val, Cp_val, grav_val, Lambda_val, &
         rho_val, P_val, T_val, opacity_val, &
         gradr_val, grada_val, gradL_val, &
         Gamma_val, gradT_val, Y_face_val, conv_vel_val, D_val, &
         mixing_type, ierr) bind(C, name="set_mlt_plain")
      use const_def, only: dp
      use auto_diff
      use turb, only: set_mlt

      ! String input (passed as length + char array from C)
      integer(c_int), intent(in), value :: mlt_option_len
      character(kind=c_char), dimension(mlt_option_len), intent(in) :: mlt_option_chars

      ! Plain scalar inputs
      real(c_double), intent(in), value :: mixing_length_alpha
      real(c_double), intent(in), value :: Henyey_MLT_nu_param
      real(c_double), intent(in), value :: Henyey_MLT_y_param
      real(c_double), intent(in), value :: max_conv_vel
      real(c_double), intent(in), value :: chiT_val, chiRho_val, Cp_val, grav_val, Lambda_val
      real(c_double), intent(in), value :: rho_val, P_val, T_val, opacity_val
      real(c_double), intent(in), value :: gradr_val, grada_val, gradL_val

      ! Plain scalar outputs (extracted %val from auto_diff outputs)
      real(c_double), intent(out) :: Gamma_val, gradT_val, Y_face_val, conv_vel_val, D_val
      integer(c_int), intent(out) :: mixing_type, ierr

      ! Local auto_diff variables
      type(auto_diff_real_star_order1) :: chiT, chiRho, Cp, grav, Lambda
      type(auto_diff_real_star_order1) :: rho, P, T, opacity
      type(auto_diff_real_star_order1) :: gradr, grada, gradL
      type(auto_diff_real_star_order1) :: Gamma_ad, gradT_ad, Y_face_ad, conv_vel_ad, D_ad
      character(len=64) :: mlt_option_str
      integer :: i

      ! Convert C char array to Fortran string
      mlt_option_str = ''
      do i = 1, mlt_option_len
         mlt_option_str(i:i) = mlt_option_chars(i)
      end do

      ! Construct auto_diff inputs: set %val, zero derivatives
      chiT = 0d0;    chiT%val = chiT_val
      chiRho = 0d0;  chiRho%val = chiRho_val
      Cp = 0d0;      Cp%val = Cp_val
      grav = 0d0;    grav%val = grav_val
      Lambda = 0d0;  Lambda%val = Lambda_val
      rho = 0d0;     rho%val = rho_val
      P = 0d0;       P%val = P_val
      T = 0d0;       T%val = T_val
      opacity = 0d0; opacity%val = opacity_val
      gradr = 0d0;   gradr%val = gradr_val
      grada = 0d0;   grada%val = grada_val
      gradL = 0d0;   gradL%val = gradL_val

      ! Call MESA set_mlt
      call set_mlt(trim(mlt_option_str), mixing_length_alpha, &
                   Henyey_MLT_nu_param, Henyey_MLT_y_param, &
                   chiT, chiRho, Cp, grav, Lambda, rho, P, T, opacity, &
                   gradr, grada, gradL, &
                   Gamma_ad, gradT_ad, Y_face_ad, conv_vel_ad, D_ad, &
                   mixing_type, max_conv_vel, ierr)

      ! Extract plain values
      Gamma_val = Gamma_ad%val
      gradT_val = gradT_ad%val
      Y_face_val = Y_face_ad%val
      conv_vel_val = conv_vel_ad%val
      D_val = D_ad%val

   end subroutine set_mlt_plain

end module mlt_wrapper_mod
