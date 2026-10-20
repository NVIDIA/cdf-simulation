/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/

/**********************************************************************
*
*  File: userfunc.c
*
*  Purpose: Lets user compile own function definitions for use in
*           expressions.  Function passed pointer to coordinate array.
*           User should also provide derivative function.
*           After defining functions, add names to arrays userfunc,
*           userfunc_deriv, and userfunc_second that follow all the
*           function definitions. 
*           Names in this file don't mattter; the first function
*           in the arrays is usr1 in datafile or queries, etc.  
*           When invoked in Evolver formulas, these functions do
*           not take arguments; arguments are implicitly point
*           coordinates.
*           Example: (usr1 + usr3)/usr10.
*
*           Also has functions for handling dynamic load libraries.
*           And elliptic functions.
*/

#include "include.h"

/**************************************************************************/

/**************************************************************************
*
*  function: userfunc_init()
*
*  purpose: called once at the start of a surface to give user functions
*           a chance to initialize.
*/

void userfunc_init()
{
  /* do whatever it takes to initialize user functions */
  
  /*  Example of error reporting. 
  kb_error(2201,"Error in userfunc_init\n",RECOVERABLE);
  */
} // end userfunc_init()

/**************************************************************************
*
*  function: usr1()
*
*  purpose: value of user defined function.
*/

/* usr1 as defined here gives conformal metric for 3-sphere in 
    stereographic projection */

REAL usr1(REAL *x /* incoming parameters */)
{ REAL denom;
  
  denom = 4+x[0]*x[0]+x[1]*x[1]+x[2]*x[2]; 
  return 16/denom/denom;
} // end usr1()

/**************************************************************************
*
*  function: usr1_deriv()
*
*  purpose: value and first derivative of user defined function.
*/
REAL usr1_deriv(
  REAL *x, /* incoming parameters */
  REAL *partials /* outgoing partial derivatives */
)
{ REAL denom,cube;
  int i;

  denom = 4+x[0]*x[0]+x[1]*x[1]+x[2]*x[2]; 
  cube = denom*denom*denom;
  for ( i = 0 ; i < SDIM ; i++ )
     partials[i] = -64/cube*x[i]; 

  return 16/denom/denom;
} // end usr1_deriv()

/**************************************************************************
*
*  function: usr1_seconds()
*
*  purpose: value, first, and second derivative of user defined function.
*/
REAL usr1_seconds(
  REAL *x, /* incoming parameters */
  REAL *partials, /* outgoing partial derivatives */
  REAL **seconds /* outgoing second derivatives */
)
{ REAL denom,cube,quart;
  int i,j;

  denom = 4+x[0]*x[0]+x[1]*x[1]+x[2]*x[2]; 
  cube = denom*denom*denom;
  quart = cube*denom;
  for ( i = 0 ; i < SDIM ; i++ )
     partials[i] = -64/cube*x[i]; 
  for ( i = 0 ; i < SDIM ; i++ )
  { for ( j = 0 ; j < SDIM ; j++ )
        seconds[i][j] = 384*x[i]*x[j]/quart;
     seconds[i][i] -= 64/cube;
  }

  return 16/denom/denom;
} // end usr1_seconds()

/***************************************************************************/

/* Another example of a user function, which is a polynomial in x,y,z.     */
/* This function is referred to as usr2 in expressions.                    */

static REAL usr_poly(REAL *x /* incoming parameters */)
{ 
    return x[0]*x[0] + x[1]*x[2] + x[2]*x[2]*x[2];
}

static REAL usr_poly_grad(
  REAL *x, /* incoming parameters */
  REAL *partials /* outgoing partial derivatives */
)
{
  partials[0] = 2*x[0];
  partials[1] = x[2];
  partials[2] = x[1] + 3*x[2]*x[2];

  return x[0]*x[0] + x[1]*x[2] + x[2]*x[2]*x[2];
}

static REAL usr_poly_hess(
  REAL *x, /* incoming parameters */
  REAL *partials, /* outgoing partial derivatives */
  REAL **seconds /* outgoing second derivatives */
)
{ 
  partials[0] = 2*x[0];
  partials[1] = x[2];
  partials[2] = x[1] + 3*x[2]*x[2];

  seconds[0][0] = 2.0;
  seconds[0][1] = seconds[1][0] = 0.0;
  seconds[0][2] = seconds[2][0] = 0.0;
  seconds[1][1] = 0.0;
  seconds[1][2] = seconds[2][1] = 1.0;
  seconds[2][2] = 6*x[2];

  return x[0]*x[0] + x[1]*x[2] + x[2]*x[2]*x[2];
}

/**************************************************************************/

/* Add your functions to these arrays; this is how they will be invoked! */

REAL (*userfunc[])(REAL*) = {usr1,usr_poly};
REAL (*userfunc_deriv[])(REAL*,REAL*) = {usr1_deriv,usr_poly_grad};
REAL (*userfunc_seconds[])(REAL*,REAL*,REAL**) = {usr1_seconds,usr_poly_hess};


/*********************************************************************
**********************************************************************

             D Y N A M I C     L O A D     L I B R A R I E S

**********************************************************************
*********************************************************************/

/*********************************************************************
*
* function: load_library()
*
* purpose: Find and load dynamic library.  Searches current directory,
*              EVOLVERPATH, default library path.
*/

#ifdef WIN32
#define dlopen(name,mode)  LoadLibrary(name)
#define dlclose(handle)    FreeLibrary(handle)
#define dlsym(handle,name) GetProcAddress(handle,name)
#endif

void load_library(char *libname)
{  
#ifdef ENABLE_DLL
  int k;
  char *env;
  char path[200];
  int len;
  void *fd;

  for ( k = 0 ; k < MAX_DLL ; k++ )
     if ( dll_list[k].name == NULL ) break;

  if ( k >= MAX_DLL )
     kb_error(2202,"Too many dynamic load libraries.\n",DATAFILE_ERROR);


  env = getenv("EVOLVERPATH");

  /* try current directory first */
  strcpy(path,"./");
  strncpy(path+2,libname,sizeof(path)-2);
  while ( (fd = dlopen(path,RTLD_NOW)) == NULL)
     { /* try paths in EVOLVERPATH */
        if ( env == NULL ) break;
        len = (int)strcspn(env,ENVPATHCHAR);
        if ( len == 0 ) break;
        strncpy(path,env,len);
        path[len] = PATHCHAR;
        strncpy(path+len+1,libname,sizeof(path)-len-2);
        if ( env[len] == 0 ) env = NULL; /* end of EVOLVERPATH */
        else env += len+1;
     } 
  /* try given name */
  if ( ! fd )
  { strncpy(path,libname,sizeof(path));
    fd = dlopen(path,RTLD_NOW);
  }
  
  if ( ! fd )
  { sprintf(errmsg,"Cannot open dynamic library %s. Reason:\n",libname);
#ifndef WIN32
    strncpy(errmsg+strlen(errmsg),dlerror(),sizeof(errmsg)-strlen(errmsg)-2);
#endif   
    kb_error(2203,errmsg,DATAFILE_ERROR);
  }

  dll_list[k].name = mycalloc(1,strlen(libname)+4);
  strcpy(dll_list[k].name,libname);

  dll_list[k].handle = fd;

#else

  kb_error(2204,"This Evolver not compiled for dynamic load libraries.\n",
     DATAFILE_ERROR);
#endif

}

/*************************************************************************
*
* function: unload_libraries
*
* purpose: unload dynamic link libraries
*/

void unload_libraries()
{
  int k;

  for ( k = 0 ; k < MAX_DLL ; k++ )
     if ( dll_list[k].name )
     { myfree(dll_list[k].name);
       dll_list[k].name = NULL;
#ifdef ENABLE_DLL
       dlclose(dll_list[k].handle); 
       dll_list[k].handle = NULL;
#endif
     }
}

/*********************************************************************
*
* function: search_libraries()
*
* purpose: find function name in dynamic load libraries.
*
* return: pointer to function. NULL if not found.
*/

dll_func_type search_libraries(char *funcname)
{ 

#ifdef ENABLE_DLL
  int i;
  dll_func_type f;

  for ( i = 0 ; i < MAX_DLL ; i++ )
    if ( dll_list[i].handle ) 
     { f = (dll_func_type)dlsym(dll_list[i].handle,funcname);
        if ( f ) return f;
     }
#endif
  return NULL;
}

/****************************************************************************
                    S Y M M E T R I C    I N T E G R A L S

  Section 19.15 of NIST Handbook of Mathematical Functions, used below to
  to compute incomplete elliptic functions of parameter m greater than 1.

****************************************************************************/

/* RF(x,y,z) = (1/2)Integrate[1/Sqrt[(t+x)(t+y)(t+z)],{t,0,Infinity}] */
REAL RF(REAL x, REAL y, REAL z)
{
  REAL lambda;
  if ( fabs(x-y) + fabs(y-z) < root8machine_eps)
  { // 7th order polynomial, so 0.01 radius should give 1e-16 accuracy
    REAL a,z1,z2,z3,ee2,ee3;
    a = (x+y+z)/3;
    z1 = (a-x)/a;
    z2 = (a-y)/a;
    z3 = (a-z)/a;
    ee2 = z1*z2 + z2*z3 + z3*z1;
    ee3 = z1*z2*z3;
    return (1 - ee2/10 + ee3/14 + ee2*ee2/24 - 3*ee2*ee3/44
            - 5*ee2*ee2*ee2/208 + 3*ee3*ee3/104 + ee2*ee2*ee3/16)/sqrt(a);
  }

  lambda = sqrt(x*y) + sqrt(y*z) + sqrt(z*x);
  return RF((x+lambda)/4,(y+lambda)/4,(z+lambda)/4);

}

/* RC(x,y) = 1/2*Integrate[1/Sqrt(t+x)/(t+y),{t,0,Infinity}] */
REAL RC(REAL x, REAL y)
{
  return RF(x,y,y);
}

/* RJ(x,y,z,p) = 3/2*Integrate[1/Sqrt[(t+x)(t+y)(t+z)]/(t+p),{t,0,Infinity}] */
REAL RJ(REAL x,REAL y,REAL z,REAL p)
{
  REAL lambda,alpha,beta;

  if ( fabs(x-y) + fabs(y-z) + fabs(z-p) < sqrt(machine_eps) )
  { REAL term = sqrt(sqrt(sqrt(x*y*z*p)));
    return 1/term/term/term;
  }
  lambda = sqrt(x*y) + sqrt(y*z) + sqrt(z*x);
  alpha = p*(sqrt(x)+sqrt(y)+sqrt(z)) + sqrt(x*y*z);
  beta = sqrt(p)*(p+lambda);

  return 0.25*RJ((x+lambda)/4,(y+lambda)/4,(z+lambda)/4,(p+lambda)/4) 
    + 3*RC(alpha*alpha,beta*beta);
}

/* RD(x,y,z) = (3/2)Integrate[1/Sqrt[(t+x)(t+y)(t+z)^3],{t,0,Infinity}] 
   Note symmetric only in x,y
*/
REAL RD(REAL x, REAL y, REAL z)
{
  REAL lambda;
  REAL c;  // scaling constant to keep size in range
 
  // Trial of series http://dlmf.nist.gov/19.36#i 19.36.2.
  // Restoring symmetry by triplicating z factor
  if ( fabs(x-y) + fabs(y-z) < root8machine_eps )
  { // 7th order polynomial, so 0.01 radius should give 1e-16 accuracy.
    REAL a,z1,z2,z3,z4,z5,ee2,ee3,ee4,ee5;
    a = (x+y+z+z+z)/5;
    z1 = (a-x)/a;
    z2 = (a-y)/a;
    z3 = (a-z)/a;
    z4 = (a-z)/a;
    z5 = (a-z)/a;
    ee2 = z1*z2 + z1*z3 + z1*z4 + z1*z5 + z2*z3 + z2*z4 + z2*z5 
           + z3*z4 + z3*z5 + z4*z5;
    ee3 = z1*z2*z3 + z1*z2*z4 + z1*z2*z5 + z1*z3*z4 + z1*z3*z5
        + z1*z4*z5 + z2*z3*z4 + z2*z3*z5 + z2*z4*z5 + z3*z4*z5;
    ee4 = z1*z2*z3*z4 + z1*z2*z3*z5 + z1*z2*z4*z5 + z1*z3*z4*z5 + z2*z3*z4*z5;
    ee5 = z1*z2*z3*z4*z5;
    /*return*/
    return  (1 - 3*ee2/14 + ee3/6 + 9*ee2*ee2/88 - 3*ee4/22 - 9*ee2*ee3/52
             + 3*ee5/26 - ee2*ee2*ee2/16 + 3*ee3*ee3/40 + 3*ee2*ee4 
             + 45*ee2*ee2*ee3/272 - 9*(ee3*ee4 + ee2*ee5)/68)/a/sqrt(a);
  }

  // duplication formula 19.26.20
  lambda = sqrt(x*y) + sqrt(y*z) + sqrt(z*x);
  c = 1 + 3*lambda/(x+y+z);
  return 2*RD((x+lambda)/c,(y+lambda)/c,(z+lambda)/c)/sqrt(c*c*c) 
             + 3/(z+lambda)/sqrt(z);
  
}


/* RG(x,y,z) = 1/4/Pi*Integrate[Integrate[Sqrt[x Sin[th]^2 Cos[ph]^2 + 
       y Sin[th]^2 Sin[ph]^2 + z Cos[th]^2] Sin[th],{th,0,Pi}], {ph,0,2 Pi}]
*/
REAL RG(REAL x, REAL y, REAL z)
{ // Formula 19.21.10
  // Completely symmetric in x,y,z so choose good permutation,
  // (x-z)(y-z) <= 0.

  return (z*RF(x,y,z) - 1/3.*(x - z)*(y - z)*RD(x,y,z) + sqrt(x*y/z))/2;
}

// From 19.25.7 and 19.21.10, combined and reduced
REAL incompleteEllipticE(REAL phi,REAL m)
{ REAL c,reduced_phi,period,value,lambda;

  if ( m == 0 ) return phi;
  if ( m < 0 ) return (incompleteEllipticE(M_PI/2+phi,-m/(1-m))
                          - ellipticE(-m/(1-m)))*sqrt(1-m);
  if ( m >= 1 && fabs(phi) > M_PI )
  { sprintf(errmsg,"incompleteEllipticE: phi %f too large for given m %f.\n",
        phi,m);
    kb_error(2663,errmsg,RECOVERABLE);
  }

  // figure period
  period = floor( (phi+M_PI/2)/M_PI );
  reduced_phi = phi - period*M_PI;
 
  if ( fabs(reduced_phi) < 1e-4 )
    value = reduced_phi - m/6*reduced_phi*reduced_phi*reduced_phi
        +(m/30 - m*m*m/40)*reduced_phi*reduced_phi*reduced_phi*reduced_phi*reduced_phi;
  else
  {
    c = 1/sin(reduced_phi);
    c *= c;
    if ( c-m < 0.0 )
    { sprintf(errmsg,"incompleteEllipticE: phi %f too large for given m %f.\n",
         phi,m);
      kb_error(3039,errmsg,RECOVERABLE);
    }
    lambda = 1/(fabs(c-1)+fabs(c-m)+fabs(c)); // get in nice range
    value = sqrt(lambda)*RF((c-1)*lambda,(c-m)*lambda,c*lambda) 
             - lambda*sqrt(lambda)*m/3*RD((c-1)*lambda,(c-m)*lambda,c*lambda);
    if ( reduced_phi < 0.0 )
      value = -value;
  }
  return value + (period ? period*2*ellipticE(m) : 0.0);
}

/****************************************************************************
                     E L L I P T I C   F U N C T I O N S
****************************************************************************/

/*****************************
 Complete elliptic integrals
*****************************/

REAL ellipticK(REAL m)
{ REAL a,b,anext;

  if ( m >= 1.0 )
     kb_error(2422,"ellipticK domain violation, parameter >= 1.\n",RECOVERABLE);

  a = 1.0;
  b = sqrt(sqrt(1 - m));

  while ( fabs(a-b) > machine_eps )
  { anext = (a + b)/2;
    b = sqrt(sqrt(a*b*(a*a+b*b)/2));
    a = anext;
  }

  return M_PI/2/a/a;
} // end ellipticK(REAL m)


REAL ellipticE(REAL m)
{ REAL a,b,anext;
  REAL K,sum = 0;
  REAL ff = 1.0;

  if ( m > 1.0 )
     kb_error(2423,"ellipticE domain violation, parameter > 1.\n",RECOVERABLE);
  if ( m == 1.0 ) return 1.0;

  a = 1.0; 
  b = sqrt(sqrt(1 - m));

  while ( fabs(a-b) > machine_eps )
  { REAL aa = a*a, bb = b*b;
    sum += ff*(aa*aa - (aa+bb)*(aa+bb)/4);
    ff *= 4;
    anext = (a + b)/2;
    b = sqrt(sqrt(a*b*(a*a+b*b)/2));
    a = anext;
  }

  K = M_PI/2/a/a;
  return K*(1.0 - sum);
} // end ellipticE()

// derivative of ellipticE
REAL ellipticEdm(REAL m)
{ if ( m == 1.0 ) return 1.0e31;
  return m==0 ? -M_PI/8 : (ellipticE(m) - ellipticK(m))/2/m;
}

// derivative of ellipticK
REAL ellipticKdm(REAL m)
{ return m==0 ? M_PI/8 : (ellipticE(m) - (1-m)*ellipticK(m))/2/m/(1-m);
}

// second derivative of ellipticE
REAL ellipticEdmdm(REAL m)
{ return  (m==0) ? -3./64*M_PI : ((m-2)*ellipticE(m) - 2*(m-1)*ellipticK(m))
            /4/m/m/(1-m);
}

// second derivative of ellipticK
REAL ellipticKdmdm(REAL m)
{ return (m == 0) ? 9./64*M_PI : 
     ((4*m-2)*ellipticE(m) + (2-5*m+3*m*m)*ellipticK(m))/4/m/m/(1-m)/(1-m);
}

// derivative of incompleteEllipticF with respect to phi
REAL incompleteEllipticFdphi(REAL phi,REAL m)
{ return 1/sqrt(1 - m*sin(phi)*sin(phi));
}


/* following Abramowitz and Stegun 17.6; replaced by symmetric form above */
REAL incompleteEllipticExx(REAL phi,REAL m)
{ REAL p,tanp,a,b,c,poweroftwo,csum,E,csinphisum,F,K,retval;
  REAL anext,bnext,cnext;

  if ( m > 1.0 )
     kb_error(2424,"incompleteEllipticE domain violation, parameter > 1.\n",
        RECOVERABLE);
  if ( m == 0 ) return phi;
  if ( m < 0 ) return -(incompleteEllipticE(M_PI/2+phi,-m/(1-m))
                          - ellipticE(-m/(1-m)))*sqrt(1-m);

  p = phi; 
  tanp = tan(p);
  a = 1.0; 
  b = sqrt(1-m);
  c = sqrt(m);
  poweroftwo = 1.0;
  csum = c*c;
  csinphisum = 0;

  while ( c > machine_eps )
  { 
    p = 2*p + atan((b/a - 1)*tanp/(1+b/a*tanp*tanp));
    tanp = (1+b/a)*tanp/(1-b/a*tanp*tanp);

    anext = (a+b)/2;
    bnext = sqrt(a*b);
    cnext = (a-b)/2;
    a = anext; b = bnext; c = cnext;

    poweroftwo *= 2;
    csum += poweroftwo*c*c;
    csinphisum += c*sin(p);
  }
  K = M_PI/2/a;
  E = K - csum*K/2;
  F = p/poweroftwo/a;
  retval = E/K*F + csinphisum;

  return retval;
} // end incompleteEllipticE()

REAL incompleteEllipticF(REAL phi, REAL m)
{ REAL p,tanp,a,b,c,poweroftwo,csum,csinphisum,F;
  REAL anext,bnext,cnext;

  if ( m == 0 ) return phi;
  if ( m < 0 ) return (incompleteEllipticF(M_PI/2+phi,-m/(1-m))
                          - ellipticK(-m/(1-m)))/sqrt(1-m);
  if ( sqrt(m)*sin(phi) > 1.0 )
     kb_error(2425,"incompleteEllipticF domain violation, m*sin(phi)^2 > 1.\n",
        RECOVERABLE);
  if ( m > 1.0 )
     return incompleteEllipticF(asin(sqrt(m)*sin(phi)),1/m)/sqrt(m);

  p = phi; 
  tanp = tan(p);
  a = 1.0; 
  b = sqrt(1-m);
  c = sqrt(m);
  poweroftwo = 1.0;
  csum = c*c;
  csinphisum = 0;

  while ( c > machine_eps )
  { 
    p = 2*p + atan((b/a - 1)*tanp/(1+b/a*tanp*tanp));
    tanp = (1+b/a)*tanp/(1-b/a*tanp*tanp);

    anext = (a+b)/2;
    bnext = sqrt(a*b);
    cnext = (a-b)/2;
    a = anext; b = bnext; c = cnext;

    poweroftwo *= 2;
    csum += poweroftwo*c*c;
    csinphisum += c*sin(p);
  }
  F = p/poweroftwo/a;

  return F;
} // end incompleteEllipticF()

// phi derivative of incompleteEllipticE
REAL incompleteEllipticEdphi(REAL phi, REAL m)
{ return sqrt(1 - m*sin(phi)*sin(phi));
}

// m derivative of incompleteEllipticE
REAL incompleteEllipticEdm(REAL phi,REAL m)
{ if ( m == 0 )
     return -(2*phi-sin(2*phi))/8;
  return (incompleteEllipticE(phi,m)-incompleteEllipticF(phi,m))/2/m;
}

// m derivative of incompleteEllipticF
REAL incompleteEllipticFdm(REAL phi,REAL m)
{ if ( m == 0 )
     return (2*phi-sin(2*phi))/8;
  return incompleteEllipticE(phi,m)/2/(m-1)/m
          - incompleteEllipticF(phi,m)/2/m
              + sin(2*phi)/4/(m-1)/sqrt(1-m*sin(phi)*sin(phi));
} // end incompleteEllipticFdm()

// value, all first, and all second derivatives of incompleteEllipticE
REAL incompleteEllipticEseconds(
  REAL phi, REAL m, /* input */
  REAL *dphi, REAL *dm,REAL *ddphi,REAL *ddm,REAL *dphidm  /* output */
    )
{ REAL E,F;
  REAL s = sin(phi);
  REAL s2 = sin(2*phi);
  REAL d = sqrt(1-m*sin(phi)*sin(phi));
  E = incompleteEllipticE(phi,m);
  F = incompleteEllipticF(phi,m);
  *dphi = d;
  if ( m == 0 )
  { *dm = -(2*phi-s2)/8;
    *ddm = -1./128*(12*phi-8*s2+sin(4*phi));
  }
  else
  { *dm   = (E - F)/2/m;
    *ddm =  -1./8/(m-1)/m/m*(2*(m-2)*E - 4*(m-1)*F + m*s2/d);
          
  }
  *ddphi = -m*cos(phi)*s/d;
  *dphidm = -s*s/2/d;
  return E;
} // end incompleteEllipticEseconds()

// value, all first, and all second derivatives of incompleteEllipticF
REAL incompleteEllipticFseconds(
  REAL phi, REAL m, /* input */
  REAL *dphi,REAL *dm,REAL *ddphi,REAL *ddm,REAL *dphidm  /* output */
  )
{ REAL E,F;
  REAL s = sin(phi);
  REAL s2 = sin(2*phi);
  REAL d = sqrt(1-m*sin(phi)*sin(phi));
  E = incompleteEllipticE(phi,m);
  F = incompleteEllipticF(phi,m);
  *dphi = 1/d;
  if ( m == 0 )
  { *dm = (2*phi-s2)/8;
    *ddm = 3./128*(12*phi-8*s2+sin(2*phi));
  }
  else
  { *dm   = -E/2/(m-1)/m - F/2/m + s2/4/(m-1)/d;
    *ddm = E/2/(m-1)/m/m + E/2/(m-1)/(m-1)/m
          - (E-F)/4/(m-1)/m/m + F/2/m/m + s*s*s2/8/(m-1)/d/d/d
          - s2/4/(m-1)/(m-1)/d + E/4/(m-1)/m/m
          + F/4/m/m - s2/8/m/(m-1)/d;
  }
  *ddphi = m*cos(phi)*s/d/d/d;
  *dphidm = s*s/2/d/d/d;
  return F;
} // end incompleteEllipticFseconds(


