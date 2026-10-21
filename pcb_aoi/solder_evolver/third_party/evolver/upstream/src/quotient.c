 /*************************************************************
 *  This file is part of the Surface Evolver source code.     *
 *  Programmer:  Ken Brakke, brakke@susqu.edu                 * 
 *************************************************************/

 /*************************************************************
 *
 *     file:     quotient.c
 *
 *     Purpose:  This file implements symmetry groups.
 *           At the moment, it has one group for rotations in xy-plane.
 *           See the files torus.c, khyp.c, etc for other examples.
 *           Four functions should be implemented for each group.
 *           These routines get added to the table in registry.c
 *           Group elements (wraps) are encoded any way you like
 *           into an integer "wrap".
 *           One function applies the group element to a point,
 *           another composes two elements, another takes inverses,
 *           and the last pulls back a one-form under the group action.
 *
 *           NOTE: The compose function needs special care in the
 *           order of composition for nonabelian groups.  The
 *           compose function is called to find the net wrap along
 *           consecutive edges, with the first edge wrap being the
 *           first argument.  If the first edge goes from U to V
 *           with wrap A, and the second goes from V to W with wrap B,
 *           then the wrapped edges are (U,AV) and (V,BW), so
 *           the wrapped triplet is (U,AV,ABW).  So compose(A,B)
 *           should return AB.
 *
 *************************************************************/


 /*************************************************************
 *     Symmetry group of rotations in the xy-plane.
 *       Uses parameter "rotation_order" for the order of the group.
 *       wrap is just an integer mod this order
 *      By John Sullivan.
 *
 *************************************************************/

#include "include.h"

/* by default the order will be four */
int rotorder = 4, genpower = 1;
static REAL rotmat[2][2] = {{0.,-1.},{1.,0.}};

/* name and pointer to the global user variable */
#define ROTORDER_NAME "rotation_order"
#define GENPOWER_NAME "generator_power"
int rotorder_var = -1;  /* reset at start of surface */
int genpower_var = -1;  /* reset at start of surface */

/* handle arithmetic mod rotorder nicely */
#define stdwrap(w) \
     ( (( ((w)<0? -(rotorder-1)*(w) : (w)) \
               + (rotorder/2)) % rotorder) - (rotorder/2) )

 /* prototypes */
 void rot_new_order(void);
 void new_rot_order(void);
 void rot_wrap(REAL*,REAL*,long);
 WRAPTYPE rot_compose(WRAPTYPE,WRAPTYPE);
 WRAPTYPE rot_inverse(WRAPTYPE);
 void rot_form_pullback( REAL *,REAL *,REAL *,WRAPTYPE);
 void frot_form_pullback( REAL *,REAL *,REAL *,WRAPTYPE);
 void pgcube_form_pullback( REAL *,REAL *,REAL *,WRAPTYPE);
 WRAPTYPE pgcube_compose(WRAPTYPE,WRAPTYPE); /* note reverse order */
 WRAPTYPE pg_tr_compose(WRAPTYPE,WRAPTYPE);
 WRAPTYPE cubel_compose(WRAPTYPE,WRAPTYPE); 
 void cubel_form_pullback( REAL *,REAL *,REAL *,WRAPTYPE);
 void xyz_wrap(REAL*,REAL*,long);
 WRAPTYPE xyz_compose(WRAPTYPE,WRAPTYPE);
 WRAPTYPE xyz_inverse( WRAPTYPE );
 void xyz_form_pullback( REAL *, REAL *, REAL *, WRAPTYPE );
 void central_wrap( REAL *, REAL *, WRAPTYPE );
 WRAPTYPE central_compose( WRAPTYPE, WRAPTYPE);
 WRAPTYPE central_inverse( WRAPTYPE );
 void central_form_pullback( REAL *, REAL *, REAL *, WRAPTYPE );
 void screw_wrap( REAL *, REAL *, WRAPTYPE );
 WRAPTYPE screw_compose(WRAPTYPE,WRAPTYPE);
 WRAPTYPE screw_inverse( WRAPTYPE );
 void screw_form_pullback( REAL *, REAL *, REAL *, WRAPTYPE );
 
 /* this function simply checks to see if the user has changed the order,
     and if so, updates the matrix */
 void new_rot_order()
 {
   if ((rotorder_var = lookup_global(ROTORDER_NAME)) < 0)
   {
     rotorder_var = add_global(ROTORDER_NAME);
     globals(rotorder_var)->value.real = 4.;
     globals(rotorder_var)->flags = ORDINARY_PARAM | RECALC_PARAMETER | ALWAYS_RECALC;
   }
   rotorder = (int)globals(rotorder_var)->value.real;
   if ((genpower_var = lookup_global(GENPOWER_NAME)) < 0)
   {
     genpower_var = add_global(GENPOWER_NAME);
     globals(genpower_var)->value.real = 1.;
     globals(genpower_var)->flags = ORDINARY_PARAM | RECALC_PARAMETER | ALWAYS_RECALC;
   }
   genpower = (int)globals(genpower_var)->value.real;

   rotmat[0][0] = rotmat[1][1] = cos(2*M_PI*genpower/rotorder);
   rotmat[1][0] = -(rotmat[0][1] = sin(2*M_PI*genpower/rotorder));
 } // end new_rot_order()

 /* called by recalc() even if not using symmetry, hence don't set user var */
 void reset_rot_order()
 {
   if (lookup_global(ROTORDER_NAME) >= 0)
   new_rot_order();
 } // end reset_rot_order()

 /* this is called so often if shouldn't do any work unless never used before */
 void rot_new_order()
 {
   if (rotorder_var<0)
   new_rot_order();
 } // end rot_new_order()

 /*******************************************************************
 *
 *  function: rot_wrap
 *
 *  purpose:  Provide adjusted coordinates for vertices that get
 *        wrapped under the group.  Serves as example for user-written
 *        symmetry function.
 *
 */

 void rot_wrap(
   REAL *x,   /* original coordinates */
   REAL *y,   /* wrapped coordinates  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   int i,j,k;
   REAL t[MAXCOORD];

   /* copy original coordinates */
   memcpy((char*)y,(char*)x,SDIM*sizeof(REAL));
   if (!wrap) return;  /* 0 should always be used to encode the identity */

   rot_new_order();  /* check that we're using the right order */
   if ( 0 == (wrap = stdwrap(wrap))) 
      return; /* adjust to have small abs value */

   /* adjust for wrapping */
   for ( i = 0 ; i < (wrap>0?wrap:-wrap); i++ )
   { memcpy((char*)t,(char*)y,SDIM*sizeof(REAL));
     /* copy in all coords, then adjust x,y */
     for (j=0; j<2; j++)
     {
       t[j]=0.;
       for (k=0;k<2;k++)
       if (wrap>0)
         t[j] += y[k]*rotmat[j][k];
       else
         t[j] += y[k]*rotmat[k][j];
     }
     memcpy((char*)y,(char*)t,SDIM*sizeof(REAL));
   }
 } // end rot_wrap()

 /********************************************************************
 *
 *  function: rot_compose()
 *
 *  purpose:  do composition of two group elements
 *
 */

 WRAPTYPE rot_compose(
   WRAPTYPE wrap1, WRAPTYPE wrap2  /* the elements to compose */
 )
 {
   rot_new_order();
   return stdwrap(wrap1 + wrap2);
 } // end rot_compose()


 /********************************************************************
 *
 *  function: rot_inverse()
 *
 *  purpose:  return inverse of group element.
 *
 */

 WRAPTYPE rot_inverse(WRAPTYPE wrap  /* the element to invert */)
 {
   rot_new_order();
   return stdwrap(-wrap);
 } // end rot_inverse()

 /*******************************************************************
 *
 *  function: rot_form_pullback
 *
 *  purpose:  Pull back differential forms at vertices that get
 *        wrapped under the group.  Serves as example for user-written
 *        symmetry function.
 *
 */

 void rot_form_pullback(
   REAL *x,   /* original coordinates */
   REAL *xform, /* result pullback */
   REAL *yform,   /* original form input  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   rot_new_order();
   rot_wrap(yform,xform,-wrap);
 } // end rot_form_pullback()

 /************************************************************************* 
 * frot is a similar group but with a flip
 * (z -> -z) with every odd rotation
 * This uses the same rotorder variable (which better be even)
 * By John Sullivan, October, 1995
 **************************************************************************/

 /* prototype */
 void frot_wrap(REAL *,REAL *,WRAPTYPE);

 /*******************************************************************
 *  function: frot_wrap
 */

 void frot_wrap(
   REAL *x,   /* original coordinates */
   REAL *y,   /* wrapped coordinates  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   int i,j,k;
   REAL t[MAXCOORD];

   /* copy original coordinates */
   memcpy((char*)y,(char*)x,SDIM*sizeof(REAL));
   if (!wrap) return;  /* 0 should always be used to encode the identity */

   rot_new_order();  /* check that we're using the right order */
   if (rotorder&1)
     kb_error(1618,"Can't use flip_rotations with odd order\n",COMMAND_ERROR);

   if ( 0 == (wrap = stdwrap(wrap))) return;  /* adjust to have small abs value */

   /* adjust for wrapping */
   for ( i = 0 ; i < (wrap>0?wrap:-wrap); i++ )
   {
     memcpy((char*)t,(char*)y,SDIM*sizeof(REAL));
     /* copy in all coords, then adjust x,y */
     for (j=0; j<2; j++)
     {
       t[j]=0.;
       for (k=0;k<2;k++)
       if (wrap>0)
         t[j] += y[k]*rotmat[j][k];
       else
         t[j] += y[k]*rotmat[k][j];
     }
     memcpy((char*)y,(char*)t,SDIM*sizeof(REAL));
     if (wrap&1) y[2] = -y[2]; /* the flip */
   }
 } // end frot_wrap()

 /*******************************************************************
 *
 *  function: frot_form_pullback
 *
 *  purpose:  Pull back differential forms at vertices that get
 *        wrapped under the group.  Serves as example for user-written
 *        symmetry function.
 *
 */

 void frot_form_pullback(
   REAL *x,   /* original coordinates */
   REAL *xform,  /* result pullback */
   REAL *yform,   /* original form input  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   frot_wrap(yform,xform,-wrap);
 } // end frot_form_pullback()

 /************************************************************************* 
 * pgcube is the full point group of a cube
 * That is, all permutations and sign changes of the coordinates.
 * By John Sullivan, December, 1995
 *
 * Wrap encoding: wrap&{1,2,4} give sign changes for x,y,z;
 * (wrap&24)/8 is power of (xyz) to cycle
 * (wrap&32)/32 tells whether to then swap x,y
 **************************************************************************/

#define pgcube_wr(signs,perm) (((perm)<<3)+(signs))
#define pgcube_perm(wrap)     (((wrap)>>3)&7)
#define pgcube_sgns(wrap)     ((wrap)&7)
#define pgcube_invperm(perm)  ( (((perm)&4) || (!(perm)))? (perm) : 3-(perm) )

 /* prototypes */
 void pgcube_wrap(REAL *,REAL *,WRAPTYPE);
 int pgcube_compperm(int,int);  /* compose two elts of S3 */
 WRAPTYPE pgcube_inverse(WRAPTYPE);
 int pgcube_permutesigns(int, int);

 /*******************************************************************
 *  function: pgcube_wrap
 */

 void pgcube_wrap(
   REAL *x,   /* original coordinates */
   REAL *y,   /* wrapped coordinates  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   int i,j;
   REAL yy;

   /* copy original coordinates */
   memcpy((char*)y,(char*)x,SDIM*sizeof(REAL));
   if (!wrap) return;  /* 0 should always be used to encode the identity */

   /* adjust for wrapping */
   for ( i = 0 ; i < 3; i++, wrap >>= 1 )
       y[i] = x[i] * ((wrap&1)? -1:1);
   for ( j = 0 ; j < (wrap&3); j++ )
   {
       yy = y[2]; y[2] = y[1]; y[1] = y[0]; y[0] = yy;
   }
   wrap >>= 2;
   if (wrap&1)
   {
       yy = y[1]; y[1] = y[0]; y[0] = yy;
   }
 } // end pgcube_wrap()

 /**************************************
 *
 *   function: pgcube_permutesigns
 *
 *   purpose:  use an S3 element to permute the bits of 3-bit number
 *
 */

 static  REAL xxx[3] = {1.,1.,1.};
 int pgcube_permutesigns(
   int signs, 
   int perm
 )
 { int retval;
   REAL  y[3];
   pgcube_wrap(xxx,y,pgcube_wr(signs,perm));
   retval = (y[0]<0.0) + ((y[1]<0.0) << 1) + ((y[2]<0.0) << 2);
   return retval;
 } // end pgcube_permutesigns()

 /********************************************************************
 *
 *  function: pgcube_compperm()
 *
 *  purpose:  compose two elts of S3
 *
 */

 int pgcube_compperm(  
   int perm1,
   int perm2
 )
 {
   int flip1,flip2,rot;
   flip1=perm1&4; flip2=perm2&4;
   rot = (perm2&3) + (flip2?-1:1)*(perm1&3);
   rot = (rot<0)? rot+3 : ( (rot>2)? rot-3:rot );
   return (flip1^flip2) + rot;
 } // end pgcube_compperm()

 /********************************************************************
 *
 *  function: pgcube_compose()
 *
 *  purpose:  do composition of two group elements
 *
 */

 WRAPTYPE pgcube_compose( 
   WRAPTYPE wrap1,
   WRAPTYPE wrap2  /* the elements to compose */
 )
 {
   int sgns1,sgns2,perm1,perm2,comp,inv,xxx,yyy;
   sgns1 = pgcube_sgns(wrap1); perm1 = pgcube_perm(wrap1);
   sgns2 = pgcube_sgns(wrap2); perm2 = pgcube_perm(wrap2);
   comp = pgcube_compperm(perm1,perm2); 
   inv = pgcube_invperm(perm1);
   xxx =  pgcube_permutesigns(sgns2,inv);
   yyy = pgcube_wr(sgns1 ^ xxx,comp);
   return yyy;
 } // end pgcube_compose()

 /********************************************************************
 *
 *  function: pgcube_inverse()
 *
 *  purpose:  return inverse of group element.
 *
 */

 WRAPTYPE pgcube_inverse(WRAPTYPE wrap  /* the element to invert */)
 {
   int sgns, perm;
   WRAPTYPE result;
   
   sgns = pgcube_sgns(wrap); perm = pgcube_perm(wrap);
   result = pgcube_wr(pgcube_permutesigns(sgns,perm),
            pgcube_invperm(perm));
   return result;
 } // end pgcube_inverse()


 /*******************************************************************
 *
 *  function: pgcube_form_pullback
 *
 *  purpose:  Pull back differential forms at vertices that get
 *        wrapped under the group.  Serves as example for user-written
 *        symmetry function.
 *
 */

 void pgcube_form_pullback(
   REAL *x,   /* original coordinates */
   REAL *xform,  /* result pullback */
   REAL *yform,   /* original form input  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   pgcube_wrap(yform,xform,pgcube_inverse(wrap));
 } // end pgcube_form_pullback()

 /************************************************************************* 
 * cubel is the full symmetry group of the unit cubic lattice
 * That is, all permutations and sign changes of the coordinates,
 * together with adding or subtracting integers.
 * By John Sullivan, December, 1995
 *
 * Wrap encoding: wrap&{1,2,4} give sign changes for x,y,z;
 * (wrap&24)/8 is power of (xyz) to cycle
 * (wrap&32)/32 tells whether to then swap x,y
 * That is, wrap&63 is the same as for pgcube.
 * Then tr=wrap/64 is a torus wrap as in torus.c: three six-bit fields;
 * using TWRAPBITS (=6) and WRAPMASK (=037) from extern.h
 * Translation is to be applied _after_ other operations.
 **************************************************************************/

#define cubel_pg(w) ((w)&077)
#define cubel_tr(w) ((w)>>6)
#define cubel_wr(tr,pg) (( ((tr)&ALLWRAPMASK) <<6)+(pg))
#define WRAPNUM(x) ( (x)>(1<<(TWRAPBITS-2)) ? (x)-(1<<(TWRAPBITS-1)) : (x))

 /* prototypes */
 void cubel_wrap(REAL *,REAL *,WRAPTYPE);
 WRAPTYPE cubel_inverse(WRAPTYPE);

 /*******************************************************************
 *  function: cubel_wrap
 */
 void cubel_wrap(
   REAL *x,   /* original coordinates */
   REAL *y,   /* wrapped coordinates  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   int i,j;

   pgcube_wrap(x,y,cubel_pg(wrap)); 
   wrap = cubel_tr(wrap);
   for (i=0; i<3; i++, wrap >>= TWRAPBITS) 
   { int w = WRAPNUM(wrap & WRAPMASK);
     if ( web.torus_period )
       for ( j = 0 ; j < SDIM ; j++ )
         y[j] += w*web.torus_period[i][j];
     else
       y[i] += w;
   }
 } // end cubel_wrap()

 /********************************************************************
 *
 *  function: pg_tr_compose()
 *
 *  purpose:  apply a cube pointgroup element to the translation 
 *  part of another wrap
 *
 */

 WRAPTYPE pg_tr_compose(
   WRAPTYPE pg,
   WRAPTYPE tr
 )
 {
   WRAPTYPE wrap=0;
   int i;
   REAL x[3],y[3];
   for (i=0; i<3; i++, tr >>= TWRAPBITS) {
     x[i] = WRAPNUM(tr & WRAPMASK);
   }
   pgcube_wrap(x,y,pg);
   for (i=2; i>=0; i--) {
     wrap = (wrap<<TWRAPBITS)+(WRAPMASK&(int)y[i]);
   }
   return wrap;
 } // end pg_tr_compose()

 /********************************************************************
 *
 *  function: cubel_compose()
 *
 *  purpose:  do composition of two group elements
 *
 */

 WRAPTYPE cubel_compose( 
   WRAPTYPE wrap1,
   WRAPTYPE wrap2  /* the elements to compose */
 )
 {
   int pg1, pg2, tr1, tr2;
   pg1 = cubel_pg(wrap1); tr1 = cubel_tr(wrap1);
   pg2 = cubel_pg(wrap2); tr2 = cubel_tr(wrap2);
   return cubel_wr( pg_tr_compose(pg1,tr2)+tr1,
           pgcube_compose(pg1,pg2) ) ;
 } // end cubel_compose()

 /********************************************************************
 *
 *  function: cubel_inverse()
 *
 *  purpose:  do inverse of group element
 *
 */

 WRAPTYPE cubel_inverse(WRAPTYPE wrap)
 {
   WRAPTYPE pg = pgcube_inverse(cubel_pg(wrap));
   WRAPTYPE tr = cubel_tr(wrap);
   return cubel_wr( torus_inverse(pg_tr_compose(pg,tr)), pg );
 } // end cubel_inverse()

 /*******************************************************************
 *
 *  function: cubel_form_pullback
 *
 *  purpose:  Pull back differential forms at vertices that get
 *        wrapped under the group.  Only pointgroup part matters.
 */

 void cubel_form_pullback(
   REAL *x,   /* original coordinates */
   REAL *xform,  /* result pullback */
   REAL *yform,   /* original form input  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 {
   pgcube_wrap(yform,xform,pgcube_inverse(cubel_pg(wrap)));
 } // end cubel_form_pullback()

 /************************************************************
 *  xyz rotation-----really is a subgp of pgcube
 *  implemented separately only for expository purposes
 *  (see Brakke and Sullivan: Using Symmetry Features of the Evolver...)
 ************************************************************/
#define xstdwrap(w) \
     ( (( ((w)<0? -2*(w):(w)) + 1) % 3) - 1 )


 void xyz_wrap(
   REAL *x, REAL *y, WRAPTYPE wrap
 )
 {
   memcpy((char*)y,(char*)x,SDIM*sizeof(REAL));
   if ( 0 == (wrap = xstdwrap(wrap))) return;

   if (wrap==-1) y[0]=x[1], y[1]=x[2], y[2]=x[0];
   else /*wrap 1*/ y[0]=x[2], y[1]=x[0], y[2]=x[1];
 } // end xyz_wrap()

 WRAPTYPE xyz_compose(WRAPTYPE wrap1,WRAPTYPE wrap2)
 { return xstdwrap(wrap1 + wrap2); }

 WRAPTYPE xyz_inverse(WRAPTYPE wrap)
 { return xstdwrap(-wrap); }

 void xyz_form_pullback(
   REAL *x, REAL *xform, REAL *yform, WRAPTYPE wrap
 )
 { xyz_wrap(yform,xform,-wrap); }

 /****************************************************************
 *
 * Central symmetry, just X <--> -X
 * Encoding: 0 is identity, 1 is inversion
 */

 void central_wrap(
   REAL *x,   /* original coordinates */
   REAL *y,   /* wrapped coordinates  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 { int n;
   for ( n = 0 ; n < SDIM ; n++ ) y[n] = wrap ? -x[n] : x[n];
 } // end central_wrap()

 WRAPTYPE central_compose(WRAPTYPE wrap1, WRAPTYPE wrap2)
 { return wrap1 ^ wrap2; }

 WRAPTYPE central_inverse(WRAPTYPE wrap)
 { return wrap; }

 void central_form_pullback(
   REAL *x, REAL *xform, REAL *yform, WRAPTYPE wrap
 )
 { int n;
   for ( n = 0 ; n < SDIM ; n++ ) xform[n] = wrap ? -yform[n] : yform[n];

 } // end central_form_pullback()

 /****************************************************************
 *
 * Screw symmetry, about z axis.  Free group with one generator.
 * Wrap number is order of generator.
 */

#define LIFT 4.0
#define TWIST  0.0
static int lift_var = -1;
static int twist_var = -1;
#define LIFTNAME "screw_height"
#define TWISTNAME "screw_angle"
 static long screw_stamp  = -1; /* when global variables checked */

 void screw_wrap(
   REAL *x,   /* original coordinates */
   REAL *y,   /* wrapped coordinates  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 { REAL angle,lift;
   if ( screw_stamp < reset_timestamp )
   { if ((lift_var = lookup_global(LIFTNAME)) < 0)
     { lift_var = add_global(LIFTNAME);
       globals(lift_var)->value.real = (REAL)LIFT;
       globals(lift_var)->flags = ORDINARY_PARAM | RECALC_PARAMETER | ALWAYS_RECALC;
     }
     if ((twist_var = lookup_global(TWISTNAME)) < 0)
     { twist_var = add_global(TWISTNAME);
       globals(twist_var)->value.real = (REAL)TWIST;
       globals(twist_var)->flags = ORDINARY_PARAM | RECALC_PARAMETER | ALWAYS_RECALC;
     }
     screw_stamp = reset_timestamp;
   }
   lift = globals(lift_var)->value.real;
   angle = globals(twist_var)->value.real * M_PI/180;
   y[0] = x[0]*cos(wrap*angle) - x[1]*sin(wrap*angle);
   y[1] = x[0]*sin(wrap*angle) + x[1]*cos(wrap*angle);
   y[2] = x[2] + wrap*lift;
 } // end screw_wrap()

 WRAPTYPE screw_compose(WRAPTYPE wrap1, WRAPTYPE wrap2)
 { return wrap1 + wrap2; }

 WRAPTYPE screw_inverse(WRAPTYPE wrap)
 { return -wrap; }

 void screw_form_pullback(
   REAL *x, REAL *xform, REAL *yform, WRAPTYPE wrap
 )
 { REAL angle;
   angle = globals(twist_var)->value.real * M_PI/180;
   xform[0] = yform[0]*cos(wrap*angle) + yform[1]*sin(wrap*angle);
   xform[1] = -yform[0]*sin(wrap*angle) + yform[1]*cos(wrap*angle);
   xform[2] = yform[2];
 } // end screw_form_pullback()


/**************************************************************************
    Symmetry group: quarter_turn
    3D torus with quarter turn in identification of top and bottom.
    x and y periods taken to be 1.
    z period is the user-defined variable quarter_turn_period.
    Generators x,y,z. x and y as in regular torus mode. z is 
    vertical translation with quarter turn: (x,y,z)->(-y,x,z).
    Relations: x z = z y^-1,   y z = z x
    Numerical representation: as in torus, for powers of x,y,z
      with generators applied in that order.
***************************************************************************/

#define QUARTERTURNNAME "quarter_turn_period"
int quarter_turn_var;

void quarter_turn_wrap(
   REAL *x,   /* original coordinates */
   REAL *y,   /* wrapped coordinates  */
   WRAPTYPE wrap  /* encoded symmetry group element */
 )
 { REAL zper;
   int xwrap,ywrap,zwrap;
   if ( quarter_turn_var < 0 )
   { quarter_turn_var = lookup_global(QUARTERTURNNAME);
     if ( quarter_turn_var < 0)
       kb_error(4008,
        "quarter_turn_period variable not defined for quarter_turn symmetry\n",
             RECOVERABLE);
   }
   xwrap = wrap & WRAPMASK;
   if ( xwrap & 0x10 ) xwrap -= 0x20;
   ywrap = (wrap >> TWRAPBITS) & WRAPMASK;
   if ( ywrap & 0x10 ) ywrap -= 0x20;
   zwrap = (wrap >> 2*TWRAPBITS) & WRAPMASK;
   if ( zwrap & 0x10 ) zwrap -= 0x20;

   y[0] = x[0] + xwrap;
   y[1] = x[1] + ywrap;
   if ( zwrap )
   { zper = globals(quarter_turn_var)->value.real;
     y[2] = x[2] + zwrap * zper;
     zwrap = (zwrap + 20) % 4;
     if ( zwrap == 1 )
     { y[0] = -(x[1] + ywrap);
       y[1] = x[0] + xwrap;
     }
     else if ( zwrap == 2 )
     { y[0] = -(x[0] + xwrap);
       y[1] = -(x[1] + ywrap);
     }
     else if ( zwrap == 3 )
     { y[0] = x[1] + ywrap;
       y[1] = -(x[0] + xwrap);
     }
   }
   else y[2] = x[2];
} // end quarter_turn_wrap()

WRAPTYPE quarter_turn_compose(WRAPTYPE wrap1, WRAPTYPE wrap2)
{ 
  int xwrap1,ywrap1,zwrap1;
  int xwrap2,ywrap2,zwrap2;
   if ( quarter_turn_var < 0 )
   { quarter_turn_var = lookup_global(QUARTERTURNNAME);
     if ( quarter_turn_var < 0)
       kb_error(3877,
        "quarter_turn_period variable not defined for quarter_turn symmetry\n",
             RECOVERABLE);
   }
   xwrap1 = wrap1 & WRAPMASK;
   if ( xwrap1 & 0x10 ) xwrap1 -= 0x20;
   ywrap1 = (wrap1 >> TWRAPBITS) & WRAPMASK;
   if ( ywrap1 & 0x10 ) ywrap1 -= 0x20;
   zwrap1 = (wrap1 >> 2*TWRAPBITS) & WRAPMASK;
   if ( zwrap1 & 0x10 ) zwrap1 -= 0x20;

   xwrap2 = wrap2 & WRAPMASK;
   if ( xwrap2 & 0x10 ) xwrap2 -= 0x20;
   ywrap2 = (wrap2 >> TWRAPBITS) & WRAPMASK;
   if ( ywrap2 & 0x10 ) ywrap2 -= 0x20;
   zwrap2 = (wrap2 >> 2*TWRAPBITS) & WRAPMASK;
   if ( zwrap2 & 0x10 ) zwrap2 -= 0x20;

   zwrap2 = (zwrap2 + 20) % 4;
   if ( zwrap2 == 1 )
   { int temp = xwrap1;
     xwrap1 = ywrap1;
     ywrap1 = -temp;
   } 
   else if ( zwrap2 == 2 )
   { xwrap1 = -xwrap1;
     ywrap1 = -ywrap1;
   }
   else if ( zwrap2 == 3 )
   { int temp = xwrap1;
     xwrap1 = -ywrap1;
     ywrap1 = temp;
   } 

   return (((zwrap1 + zwrap2) & WRAPMASK) << (2*TWRAPBITS))
        + (((ywrap1 + ywrap2) & WRAPMASK) << TWRAPBITS)
        + ((xwrap1 + xwrap2) & WRAPMASK); 

 } // end quarter_turn_compose()

WRAPTYPE quarter_turn_inverse(WRAPTYPE wrap)
{ int temp;
  int xwrap,ywrap,zwrap;
  xwrap = wrap & WRAPMASK;
  if ( xwrap & 0x10 ) xwrap -= 0x20;
  ywrap = (wrap >> TWRAPBITS) & WRAPMASK;
  if ( ywrap & 0x10 ) ywrap -= 0x20;
  zwrap = (wrap >> 2*TWRAPBITS) & WRAPMASK;
  if ( zwrap & 0x10 ) zwrap -= 0x20;
  xwrap = -xwrap;
  ywrap = -ywrap;
  zwrap = -zwrap;
  switch ( (zwrap + 20) % 4 )
  { case 1: 
     temp = xwrap;
     xwrap = ywrap;
     ywrap = -temp;
     break;
    case 2:
     xwrap = -xwrap;
     ywrap = -ywrap;
     break;
    case 3:
     temp = xwrap;
     xwrap = -ywrap;
     ywrap = temp;
     break;
  }
  
  return ((zwrap  & WRAPMASK) << (2*TWRAPBITS))
        + ((ywrap & WRAPMASK) << TWRAPBITS)
        + (xwrap  & WRAPMASK); 

 } // end quarter_turn_inverse()

 void quarter_turn_form_pullback(
   REAL *x, REAL *xform, REAL *yform, WRAPTYPE wrap
 )
 { int zwrap;
   zwrap = (wrap >> 2*TWRAPBITS) & WRAPMASK;
   yform[2] = xform[2];
   switch ( (zwrap + 20) % 4 )
   { case 0:
        yform[0] = xform[0];
        yform[1] = xform[1];
        break;
     case 1:
        yform[0] = xform[1];
        yform[1] = -xform[0];
        break;
     case 2:
        yform[0] = -xform[0];
        yform[1] = -xform[1];
        break;
     case 3:
        yform[0] = -xform[1];
        yform[1] = xform[0];
        break;
        break;
   }
 } // end quarter_turn_form_pullback()

