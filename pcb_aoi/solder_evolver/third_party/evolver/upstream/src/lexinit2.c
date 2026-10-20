/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/

/*************************************************************
*
*     file:        lexinit2.c
*
*     Purpose:    Routines used by initialize() in lexinit.c
*                 Split because lexinit.c got too big for Mac
*
*/

#include "include.h"
#include "lex.h"
#include "ytab.h"
#define yylex() kb_yylex(NULL)

/*********************************************************************
*
* function: torus_period_init()
*
* purpose: set up torus period arrays.
*/

void torus_period_init()
{ int i;

  if ( web.torus_period ) return;

  web.torus_period = dmatrix(0,SDIM-1,0,SDIM-1);
  web.inverse_periods = dmatrix(0,SDIM-1,0,SDIM-1);
  web.inverse_periods_tr = dmatrix(0,SDIM-1,0,SDIM-1);
  set_torus_periods_global();
  set_inverse_periods_global();
  for ( i = 0 ; i < SDIM ; i++ )
    web.torus_period[i][i] = web.inverse_periods[i][i] = 1.0;
  web.torusv = 1.0;
} // end torus_period_init()

/*********************************************************************
*
* function: torus_display_period_init()
*
* purpose: set up torus period arrays.
*/

void torus_display_period_init()
{ int i;

  if ( web.torus_display_period ) return;

  web.torus_display_period = dmatrix(0,SDIM-1,0,SDIM-1);
  web.inverse_display_periods = dmatrix(0,SDIM-1,0,SDIM-1);
  for ( i = 0 ; i < SDIM ; i++ )
    web.torus_display_period[i][i] = web.inverse_display_periods[i][i] = 1.0;
} // end torus_display_period_init()

/****************************************************************
*
*  Function: read_periods()
*
*  Purpose:  Reads torus periods.
*/

void read_periods()
{
  int i,j;

  torus_period_init();

  /* read in torus periods */
  for ( i = 0 ; i < SDIM ; i++ )
    for ( j = 0 ; j < SDIM ; j++ )
    { int esize;
      esize = exparse(0,&torus_period_expr[i][j],USERCOPY);
      if ( esize <= 0 )
      { sprintf(errmsg,
            "Bad torus_period[%d][%d] definition.  Check space dimension\n",
                   i+1,j+1);
        kb_error(3903,errmsg,DATAFILE_ERROR);
        return;
      }
      sprintf(torus_period_expr[i][j].name,"torus period [%d][%d]",i,j);
    }
  calc_periods(ADJUST_VOLUMES);

} // end read_periods()

/****************************************************************
*
*  Function: read_display_periods()
*
*  Purpose:  Reads torus periods for display.
*/

void read_display_periods()
{
  int i,j;

  torus_display_period_init();

  /* read in display periods */
  for ( i = 0 ; i < SDIM ; i++ )
    for ( j = 0 ; j < SDIM ; j++ )
    { int esize;
      esize = exparse(0,&torus_display_period_expr[i][j],USERCOPY);
      if ( esize <= 0 )
      { sprintf(errmsg,
            "Bad display_period[%d][%d] definition.  Check space dimension\n",
                   i+1,j+1);
        kb_error(1670,errmsg,DATAFILE_ERROR);
        return;
      }
      sprintf(torus_display_period_expr[i][j].name,"display period [%d][%d]",i,j);
    }
  calc_periods(ADJUST_VOLUMES);
} // end read_display_periods()

/****************************************************************
*
*  Function: read_parameter()
*
*  Purpose:  Reads one parameter. 
*/

void read_parameter()
{
  int n=0;
  struct global *p;
  int oldtok = tok; /* whether optimizing */

  tok = yylex(); /* dispose of PARAMETER */
  if ( tok == IDENT_TOK )
  { p = globals(yylval.i);
    if ( !(p->flags & LEFTOVER) && !addload_flag ) 
    { sprintf(errmsg,"Redefinition of identifier '%s'.\n",yytext);
      kb_error(1671,errmsg,DATAFILE_ERROR);
    }
    p->flags &= ~LEFTOVER;
  }
  else 
  { if ( tok != NEWIDENT_TOK )
     { kb_error(1672,"Need PARAMETER identifier.\n",WARNING);
        return;
     }
     if ( strlen(yytext) < 2 )
        kb_error(1480,"Identifiers must be at least two characters long.\n",
            DATAFILE_ERROR);
     n = add_global(yytext);
     if ( n < 0 )
         kb_error(1350,"Duplicate parameter name.\n",DATAFILE_ERROR);
     p = globals(n);
  }
  p->value.real = 0.0;  /* default */
  p->type = REAL_TYPE;
  p->flags |= ORDINARY_PARAM;
  p->attr.varstuff.delta = OPTPARAM_DELTA;
  p->attr.varstuff.pscale = 1.0;
  if ( (oldtok == OPTIMIZING_PARAMETER_TOK) && !(p->flags & OPTIMIZING_PARAMETER) )
  { p->flags |= OPTIMIZING_PARAMETER;
     if ( optparamcount >= MAXOPTPARAM-1 )
        kb_error(1674,"Too many optimizing parameters. Change MAXOPTPARAM in extern.h if you really need more.\n",DATAFILE_ERROR);
     else optparam[optparamcount++].pnum = n;
  }

  tok = yylex(); /* dispose of IDENT_ */
  if ( (tok == '=') || (tok == ASSIGN_TOK ) )/* have initial value */
  { if ( read_const(&p->value.real) < 0 )
    { if ( tok == QUOTATION_TOK )
      { p->value.string = mycalloc(strlen(yytext)+2,sizeof(char));
        strcpy(p->value.string,yytext);
        p->type = STRING_TYPE;
        p->flags |= STRINGVAL;
        p->flags &= ~ORDINARY_PARAM;
        tok = yylex();  // eat string token
        tok = yylex();
      } 
      else
        kb_error(1351,"Need constant expression for initial value.\n",DATAFILE_ERROR);
    }
    else tok = yylex();  /* get back token exparse pushed back */
    if ( tok == ';' ){ verb_flag = 0; tok = yylex();  /* permit ; */ }
  }
  else if ( tok == PARAMETER_FILE_TOK )
  {
    tok = yylex();
    if ( tok == QUOTATION_TOK )
    { FILE *pfd = path_open(yytext,NOTDATAFILENAME);
      int k;
      REAL val; 

      if ( pfd == NULL )
      { 
        sprintf(errmsg, "Cannot open parameter file %s.\n",yytext);
        kb_error(1356,errmsg, DATAFILE_ERROR);
        return;
      }
      p->value.file.value_file = mycalloc(strlen(yytext)+1,1);
      strcpy(p->value.file.value_file,yytext);
      p->value.file.values = (REAL *)mycalloc(1000,sizeof(REAL));
      while ( fgets(msg,msgmax,pfd) )
      {
#ifdef FLOAT128
        k = atoi(msg);
        strtok(msg," \t");
        val = atof(strtok(NULL," \t"));
#elif defined(LONGDOUBLE)
        sscanf(msg,"%d %Lf",&k,&val);
#else
        sscanf(msg,"%d %lf",&k,&val);
#endif
        if ( k >= 1000 )
        { kb_error(1677,"Too high element number in parameter file.\n",WARNING);
          break;
        }
        p->value.file.values[k] = val;
      }
      fclose(pfd);
      p->flags |= FILE_VALUES;
      tok = yylex();
    }
    else
      kb_error(1678,"Parameter file name missing.\n",DATAFILE_ERROR);

  }
  else
  { int errline = line_no;
    unput_tok();
    if ( read_const(&p->value.real) >= 0 )      
    { sprintf(errmsg,"Missing '=' after parameter name on line %d.\n",errline);
      kb_error(2672,errmsg,WARNING);
    }
    tok = yylex();  /* get back token exparse pushed back */
    if ( tok == ';' ){ verb_flag = 0; tok = yylex();  /* permit ; */ }
  
  }

  for (;;)
  {
    if ( (tok == PDELTA_TOK) || ( stricmp(yytext,"delta") == 0) )
    { tok = yylex(); /* dispose of DELTA_TOK */
       if ( (tok == '=') || (tok == ASSIGN_TOK) ) /* have initial value */
       { if ( read_const(&p->attr.varstuff.delta) < 0 )
            kb_error(1675,"Need constant expression for pdelta.\n",DATAFILE_ERROR);
          else tok = yylex();  /* get back token exparse pushed back */
       }
    }
    else if ( tok == PSCALE_TOK || tok == SCALE_TOK)
    { tok = yylex(); /* dispose of SCALE_TOK */
      if ( (tok == '=') || (tok == ASSIGN_TOK) ) /* have initial value */
      { if ( read_const(&p->attr.varstuff.pscale) < 0 )
           kb_error(1676,"Need constant expression for scale.\n",DATAFILE_ERROR);
        else tok = yylex();  /* get back token exparse pushed back */
      }
    }
    else if ( tok == ON_ASSIGN_CALL_TOK )
    { tok = yylex(); /* dispose of ON_ASSIGN_CALL_TOK */
      if ( tok == PROCEDURE_IDENT_TOK )
      { struct global *gg = globals(yylval.i);
        if ( gg->attr.procstuff.argcount != 0 )
          kb_error(4765,"on_assign_call procedure cannot have arguments.\n",
           DATAFILE_ERROR);
        p->attr.varstuff.on_assign_call = yylval.i;
        gg->flags |= USED_ON_ASSIGN_CALL;
        tok = yylex();
      }
      else 
        kb_error(5629,"on_assign_call must be followed by command name.\n",
          DATAFILE_ERROR);
    }
    else break;
  } 
  recovery_flag = 0;
} // end read_parameter()
         
      
/*************************************************************************
*
* Reads and parses information for one free boundary specification.
* Current line starts with BOUNDARY keyword.
* Returns boundary number, or -1 in case of early error.
*/

int read_boundary()
{
  int bnum=0;  /* boundary number */
  int pcount; /* number of parameters */
  int i,k;
  int esize;
  struct boundary *bdry;

  if ( V_BOUNDARY_ATTR == 0 )
  { int one = 1;
    V_BOUNDARY_ATTR = 
	  add_attribute(VERTEX,"v_boundary",INTEGER_TYPE,0,&one,0,NULL,MPI_NO_PROPAGATE);
    E_BOUNDARY_ATTR = 
	  add_attribute(EDGE,"e_boundary",INTEGER_TYPE,0,&one,0,NULL,MPI_NO_PROPAGATE);
    F_BOUNDARY_ATTR = 
	  add_attribute(FACET,"f_boundary",INTEGER_TYPE,0,&one,0,NULL,MPI_NO_PROPAGATE);
    EXTRAS(VERTEX)[V_BOUNDARY_ATTR].flags |= READ_ONLY_ATTR;
    EXTRAS(EDGE)[E_BOUNDARY_ATTR].flags |= READ_ONLY_ATTR;
    EXTRAS(FACET)[F_BOUNDARY_ATTR].flags |= READ_ONLY_ATTR;
  }

  tok = yylex();  /* eat BOUNDARY token */
  if ( (tok != INTEGER_TOK) && (tok != NEWIDENT_TOK) && (tok != BOUNDARY_NAME_TOK) ) 
  { kb_error(1679,"Need boundary number or name.\n",DATAFILE_ERROR);
    return -1;
  }
  if ( tok == INTEGER_TOK )
  { bnum = yylval.i;     /* boundary number */
    if ( bnum < 0 ) 
    { sprintf(errmsg,"Bad boundary number: %d.\n",bnum);
      kb_error(1680,errmsg,DATAFILE_ERROR);
      tok = yylex();
      return -1;
    }
    if ( bnum >= web.bdrymax )
    { web.boundaries = (struct boundary *)kb_realloc((char*)web.boundaries,
         (bnum+10)*sizeof(struct boundary));
      web.bdrymax = bnum+10;
    }
    if ( bnum > web.highbdry )
      web.highbdry = bnum;
    if ( web.boundaries[bnum].attr & IN_USE )
    { int moved_flag = 0;
      if ( web.boundaries[bnum].attr & NAMED_THING )
      /* move named boundary */
      do
      { for ( i = 1 ; i < web.bdrymax ; i++ )
          if ( !(web.boundaries[i].attr & IN_USE) )
          { web.boundaries[i] = web.boundaries[bnum];
            memset((char*)(&web.boundaries[bnum]),0,sizeof(struct boundary));
            web.boundaries[i].num = i;
            k = lookup_global(web.boundaries[i].name);
            globals(k)->value.bnum = i;
            if ( i > web.highbdry )
              web.highbdry = i;
            moved_flag = 1;
            break;
          }
        if ( i >= web.bdrymax )   
        { web.boundaries = (struct boundary *)kb_realloc((char*)web.boundaries,
           (web.bdrymax+10)*sizeof(struct boundary));
          web.bdrymax = web.bdrymax+10;
        }
      } while ( !moved_flag );
      else if ( !(web.boundaries[bnum].attr & BDRY_FORWARD_DEF)  && !addload_flag)
      { sprintf(errmsg,"Boundary number %d already defined.\n",bnum);
        kb_error(2120,errmsg,DATAFILE_ERROR);
        tok = yylex();
        return -1; 
      }
    }
  } 
  else /* name */
  {
    for ( i = 1 ; i < web.bdrymax ; i++ )
    {
      if ( !(web.boundaries[i].attr & IN_USE) )
      { bnum = i; break; } 
      if ( stricmp(yytext,web.boundaries[i].name) == 0 )
      { if ( (datafile_flag == IN_DATAFILE) && !addload_flag 
                  && !(web.boundaries[i].attr & BDRY_FORWARD_DEF))
        { sprintf(errmsg,"Boundary name %s already used.\n",yytext);
          kb_error(2121,errmsg,DATAFILE_ERROR);
          tok = yylex();
          return -1;
        }
        else
        { bnum = i;
          break;
        }
      }
    }
    if ( i > web.highbdry )
      web.highbdry = i;
    if ( i >= web.bdrymax )
    { web.boundaries = (struct boundary *)kb_realloc((char*)web.boundaries,
           (web.bdrymax+10)*sizeof(struct boundary));
      web.bdrymax = web.bdrymax+10;
      bnum = i;
    }

    if ( tok == NEWIDENT_TOK )
    { k = add_global(yytext);
      globals(k)->flags |= BOUNDARY_NAME;
      globals(k)->value.bnum = bnum;
    }
  }
  bdry = web.boundaries + bnum;
  
  // fix memory lead for addload
  if ( bdry->coordf[0] )
  { 
    for ( i = 0 ; i < MAXCOORD ; i++ )
    { free_expr(bdry->coordf[i]);
      free_expr(bdry->convect[i]);
      free_expr(bdry->envect[i]);
    }
    myfree((char*)bdry->coordf[0]);
  }

  memset((char*)bdry,0,sizeof(struct boundary));
  bdry->num = bnum;
  bdry->attr |= IN_USE;
  if ( tok == INTEGER_TOK ) 
    sprintf(web.boundaries[bnum].name,"%d",bnum);
  else
  { strncpy(bdry->name,yytext,BDRYNAMESIZE-1);
    bdry->attr |= NAMED_THING;
  }
  
  tok = yylex();
  if ( tok == ';' )
  { bdry->attr |= BDRY_FORWARD_DEF;
    verb_flag = 0;  /* set by ';' */
    tok = yylex();
    return bnum;
  }
  if ( tok != PARAMETERS_TOK )
     { sprintf(errmsg,"Expecting PARAMETERS keyword for boundary %s.\n",bdry->name);
       kb_error(1683,errmsg,DATAFILE_ERROR);
       pcount = bdry->pcount = 1; 
       /* try to continue */
     }
  else
  { tok = gettok(INTEGER_TOK);
    bdry->pcount = pcount = yylval.i;
  }
  if ( pcount > web.maxparam )
  { web.maxparam = pcount;
    expand_attribute(VERTEX,V_PARAM_ATTR,&web.maxparam);
  }
  if ( (pcount < 0) || (tok != INTEGER_TOK) ) 
     { sprintf(errmsg,"Bad parameter count %d for boundary %s. Assuming 1.\n",
          pcount,bdry->name);
       kb_error(1684,errmsg,DATAFILE_ERROR);
       pcount = bdry->pcount = 1;  /* try to continue */
     }
  if ( pcount > MAXPARAM )
     { sprintf(errmsg,
            "Parameter count for boundary %s exceeds %d. Assuming %d.\n",
                                          bdry->name,MAXPARAM,MAXPARAM);
       kb_error(1685,errmsg,DATAFILE_ERROR);
       bdry->pcount = pcount = MAXPARAM; /* try to continue */
     }

  tok = yylex();
  for (;;)
    switch (tok )
    { case CONVEX_TOK:
       web.convex_flag = 1;
       bdry->attr |= B_CONVEX;
       tok = yylex();
       break;
      case CONTENT_RANK_TOK:
        { REAL rank;
          if ( read_const(&rank) < 0 )
            kb_error(3467,"Need integer value for content_rank.\n",
              DATAFILE_ERROR); 
          bdry->content_rank = (int)rank;
          tok = yylex();  /* lookahead */
          break;
        }
      case NONWALL_TOK:
       bdry->attr |= NONWALL;
       tok = yylex();
       break;
      case PARTNER_HITTING_TOK:
       bdry->attr |= PARTNER_HITTING;
       tok = yylex();
       break;
      default: goto read_coord_funcs;
     }


read_coord_funcs:
  /* read and parse coordinate functions */
  bdry->coordf[0] = (struct expnode *)mycalloc(SDIM+2*MAXCOORD,
                            sizeof(struct expnode));
  for ( i = 1 ; i < SDIM ; i++ )
    bdry->coordf[i] = bdry->coordf[0] + i;

  for ( i = 0 ; i < SDIM ; i++ )
  {
    if ( (tok != COORD_TOK ) || ( yylval.i != 1 + i ))
    { sprintf(errmsg,
          "Bad coordinate %d definition for boundary %s.\n",i+1,bdry->name);
      goto berr;
    }
    boundary_expr_flag = 1;
    esize = exparse(pcount,bdry->coordf[i],USERCOPY);
    boundary_expr_flag = 0;
    tok = yylex();
    if ( esize <= 0 )
    { sprintf(errmsg,
       "Bad coordinate %d definition for boundary %s.\n",i+1,bdry->name);
      goto berr;
    }
    sprintf(msg,"boundary %s component %d",bdry->name,i+1);
    strncpy(bdry->coordf[i]->name,msg,EXPNAMESIZE-1);
  }

  /* various integrands */
  for ( i = 0 ; i < MAXCOORD ; i++ )
     { bdry->envect[i] = bdry->coordf[0] + SDIM + i;
        bdry->convect[i] = bdry->coordf[0] + SDIM + MAXCOORD + i;
     }
  for (;;)
    switch ( tok )
    { 
      case ENERGY_TOK:
      /* read and parse energy function */
      tok = yylex();
      for ( i = 0 ; i < MAXCOORD ; i++ )
      {
        if ( (tolower(yytext[0]) != 'e') || (yytext[1] != '1' + i) )
            break;
        esize = exparse(SDIM,bdry->envect[i],USERCOPY);
        tok = yylex();  
        if ( esize <= 0 )
        { sprintf(errmsg,
           "Bad energy component %d definition for boundary %s.\n",
            i+1,bdry->name);
          kb_error(1367,errmsg,DATAFILE_ERROR);
          return bnum;
        }
        sprintf(msg,"boundary %s energy component %d",bdry->name,i+1);
        strncpy(bdry->envect[i]->name,msg,EXPNAMESIZE-1);
      }
      if ( i == 0 )
      { sprintf(errmsg,"Missing energy components for boundary %s\n",
             bdry->name);
        kb_error(5932,errmsg,DATAFILE_ERROR);
      }

      bdry->compcount = i;
      bdry->attr |= CON_ENERGY;
      break;

     case CONTENT_TOK:
      /* read and parse content vector potential */
      tok = yylex();
      bdry->attr |= CON_CONTENT;
      for ( i = 0 ; i < SDIM ; i++ )
      {
        if ( (tolower(yytext[0]) != 'c') || (yytext[1] != '1' + i) )
            break;
        esize = exparse(SDIM,bdry->convect[i],USERCOPY);
        tok = yylex();
        if ( esize <= 0 )
        { sprintf(errmsg,
              "Bad content component %d definition for boundary %s.\n",
                 i+1,bdry->name);
          kb_error(1378,errmsg,DATAFILE_ERROR);
          return bnum;
        }
        sprintf(msg, "boundary %s content component %d", bdry->name,i+1);
        strncpy(bdry->convect[i]->name,msg,EXPNAMESIZE-1);
      } 
      /* check consistency of number of components */
      if ( bdry->compcount )
      { if ( bdry->compcount != i )
          kb_error(1381,
             "Inconsistent number of components in content integrand.\n",
                   WARNING);
      }
      else
      { if ( (i != 1) && (i != SDIM) )
         kb_error(1449,"Illegal number of components in content integrand.\n",
                   DATAFILE_ERROR);
         bdry->compcount = i;
      }
      bdry->attr |= CON_CONTENT;
      break;
    
     default: return bnum; 
    } /* end switch for integrands */

berr:
  kb_error(1686,errmsg,DATAFILE_ERROR);
  return bnum;

} /* end read_boundary() */

/*************************************************************************
*
* Reads and parses information for one free constraint specification.
* Current line starts with constraint keyword.
* Returns constraint number.
*/

int read_constraint()
{
  int cnum=0;  /* constraint number */
  int i,k;
  int esize;
  int more_attr;
  struct constraint *con;
  int done_flag;

  tok = yylex();  /* eat CONSTRAINT */
  if ( (tok != INTEGER_TOK)  && (tok != NEWIDENT_TOK) && ( tok != CONSTRAINT_NAME_TOK) ) 
  { kb_error(1687,"Need constraint number or name.\n",DATAFILE_ERROR);
    return -1;
  }
  if ( tok == INTEGER_TOK )
  { cnum = yylval.i;     /* constraint number */
    if ( cnum < 0 ) 
     { kb_error(1688,"Constraint number must be positive.\n",DATAFILE_ERROR);
       tok = yylex();
       return -1;
     }
    while ( cnum >= web.maxcon )
    { int newcons = web.maxcon ? 2*web.maxcon : 16;
      web.constraints = (struct constraint *) kb_realloc((char*)web.constraints,
          newcons*sizeof(struct constraint));
      web.maxcon = newcons;
    }
    if ( cnum > web.highcon )
      web.highcon = cnum;
    con = get_constraint(cnum);
    if ( con->attr & IN_USE )
    { 
      if ( con->attr & NAMED_THING )
      { /* Oops. Move the named constraint */
        for ( i = 1 ;  ; i++ )
        { struct constraint *c;

          if ( i >= web.maxcon ) // need more constraint structures
          { int newcons = 2*web.maxcon;
            web.constraints = 
              (struct constraint *) kb_realloc((char*)web.constraints,
              newcons*sizeof(struct constraint));
            web.maxcon = newcons;
            con = get_constraint(cnum);  // might hvae move during re-allocation
          }
          c = get_constraint(i);
          if ( !(c->attr & IN_USE) )
          { struct global *g;
            *c = *con;
            memset((char*)con,0,sizeof(struct constraint));
            k = lookup_global(c->name);
            g = globals(k);  
            g->value.cnum = i;
            if ( i > web.highcon )
              web.highcon = i;
            break;
          }   
        }
       }           
       else if ( datafile_flag & !(con->attr & CON_FORWARD_DEF)
                         & !addload_flag )
       { sprintf(errmsg,"Constraint number %d already defined.\n",cnum);
         kb_error(1690,errmsg,DATAFILE_ERROR);
         tok = yylex();
         return -1;
       }
       else
         con->attr &= ~CON_FORWARD_DEF;
    }
  }
  else /* named constraint */
  { 
    for ( i = 1 ; i < web.maxcon ; i++ )
    { con = get_constraint(i);
      if ( !(con->attr & IN_USE) )
      { cnum = i; break; }   
      if ( stricmp(yytext,get_constraint(i)->name) == 0 ) 
      { if ( datafile_flag == IN_DATAFILE && !(con->attr & CON_FORWARD_DEF)
            && !addload_flag )
        { sprintf(errmsg,"Constraint name %s already used.\n",yytext);
          kb_error(2122,errmsg,DATAFILE_ERROR);
          tok = yylex();
          return -1;
        }
        else 
        { cnum = i;
          con->attr &= ~CON_FORWARD_DEF;
          break;
        }
      }
    }
    if ( i >= web.maxcon )
    { int newcons = web.maxcon ? 2*web.maxcon : 16;
      web.constraints = (struct constraint *) kb_realloc((char*)web.constraints,
              newcons*sizeof(struct constraint));
      cnum = web.maxcon;
      if ( cnum == 0 )
         cnum = 1;  // not using constraint number 0
      web.maxcon = newcons;
    }
    if ( i > web.highcon )
      web.highcon = i;
    if ( tok == NEWIDENT_TOK )
    { k = add_global(yytext);
      globals(k)->flags |= CONSTRAINT_NAME;
      globals(k)->value.cnum = cnum;
    }
  } // end else named constraint

  con = GETCONSTR(cnum);
  // fix memory leak for addload
  if ( con->formula )
  { free_expr(con->formula);
    for ( i = 0 ; i < MAXCONCOMP ; i++ )
    { free_expr(con->convect[i]);
      free_expr(con->envect[i]);
    }
    myfree((char*)con->formula);
  }

  memset((char*)con,0,sizeof(struct constraint));
  if ( tok == INTEGER_TOK )
    sprintf(con->name,"%d",cnum);
  else
  { strncpy(con->name,yytext,CONNAMESIZE-1);
    con->attr |= NAMED_THING;
  }
  con->attr |= IN_USE;
  tok = yylex();
  if ( tok == ';' )
  { con->attr |= CON_FORWARD_DEF;
    verb_flag = 0; /* ';' sets verb_flag */
    tok = yylex();
    return cnum;
  } 
  for ( more_attr = 1 ; more_attr ; )
    switch ( tok )
    {
      case CONTENT_RANK_TOK:
        { REAL rank;
          if ( read_const(&rank) < 0 )
            kb_error(1911,"Need integer value for content_rank.\n",
              DATAFILE_ERROR); 
          con->content_rank = (int)rank;
          tok = yylex();  /* lookahead */
          break;
        }
      case CONVEX_TOK:
          web.convex_flag = 1;
          con->attr |= B_CONVEX;
          tok = yylex();
          break;

      case NONWALL_TOK:
          con->attr |= NONWALL;
          tok = yylex();
          break;

      case NONNEGATIVE_TOK:
          web.constr_flag = 1;
          con->attr |= NONNEGATIVE;
          one_sided_present = 1;
          tok = yylex();
          break;

      case NONPOSITIVE_TOK:
          web.constr_flag = 1;
          con->attr |= NONPOSITIVE;
          one_sided_present = 1;
          tok = yylex();
          break;

      case GLOBAL_TOK:
          web.constr_flag = 1;
          con->attr |= GLOBAL;
          web.con_global_map[web.con_global_count++] = (conmap_t)cnum;
          tok = yylex();
          break;

      default: more_attr = 0;
    }
     
  if ( one_sided_present && (raw_velocity_attr < 0) )
  { int dim = SDIM;
    raw_velocity_attr = add_attribute(VERTEX,RAW_VELOCITY_ATTR_NAME,
        REAL_TYPE,1,&dim,0,NULL,MPI_PROPAGATE);
  }

  /* read and parse defining function */
  constraint_init(con);
  if ( tok != FUNCTION_TOK )
  { sprintf(errmsg,
       "Expected function definition for constraint %s.\n",con->name);
    kb_error(1691,errmsg,DATAFILE_ERROR);
    return cnum;
  }
  esize = exparse(SDIM,con->formula,USERCOPY);
  tok = yylex();
  if ( esize <= 0 )
  { sprintf(errmsg,"Bad function definition for constraint %s.\n",con->name);
    kb_error(1692,errmsg,DATAFILE_ERROR);
    return cnum;
  }
  sprintf(msg,"constraint %s formula",con->name);
  strncpy(con->formula->name,msg,EXPNAMESIZE-1);

  /* various integrands */
  done_flag = 0;
  while ( !done_flag )
    switch ( tok )
    { 
     case ENERGY_TOK:
      /* read and parse energy function */
      tok = yylex();
      for ( i = 0 ; i < MAXCONCOMP ; i++ )
      {
       if ( (tolower(yytext[0]) != 'e') || (yytext[1] != '1' + i) )
            break;
       esize = exparse(SDIM,con->envect[i],USERCOPY);
       tok = yylex();  
       if ( esize <= 0 )
       { sprintf(errmsg,
            "Bad energy component %d definition for constraint %s.\n",
               i+1,con->name);
         kb_error(1693,errmsg,DATAFILE_ERROR);
         return cnum;
       }
       sprintf(msg,"constraint %s energy component %d",con->name,i+1);
       strncpy(con->envect[i]->name,msg,EXPNAMESIZE-1);
      }
      if ( i == 0 )
      { sprintf(errmsg,"Missing energy components for constraint %s\n",
             con->name);
        kb_error(3228,errmsg,DATAFILE_ERROR);
      }
      con->compcount = i;
      con->attr |= CON_ENERGY;
      break;

     case CONTENT_TOK:
      /* read and parse content vector potential */
      tok = yylex();
      con->attr |= CON_CONTENT;
      for ( i = 0 ; i < SDIM ; i++ )
      {
        if ( (tolower(yytext[0]) != 'c') || (yytext[1] != '1' + i) )
          break;
       esize = exparse(SDIM,con->convect[i],USERCOPY);
       tok = yylex();
       if ( esize <= 0 )
       { sprintf(errmsg,
              "Bad content component %d definition for constraint %s.\n",
                  i+1,con->name);
         kb_error(1694,errmsg,DATAFILE_ERROR);
         return cnum;
       }
       sprintf(msg,"constraint %s content component %d",con->name,i+1);
       strncpy(con->convect[i]->name,msg,EXPNAMESIZE-1);
      }
      /* check consistency of number of components */
      if ( con->compcount )
      { if ( con->compcount != i )
          kb_error(1695,
            "Inconsistent number of components in content integrand.\n",
               WARNING);
      }
      else
      { if ( (i != 1) && (i != SDIM) )
          kb_error(1696,"Illegal number of components in content integrand.\n",
                  DATAFILE_ERROR);
        con->compcount = i;
      }
      con->attr |= CON_CONTENT;
      break;

     default: done_flag = 1; break; 
    } /* end switch for integrands */

  if ( everything_quantities_flag )
    convert_constraint_to_quantities(cnum);
  return cnum;
} // end read_constraint()


/*************************************************************************
*
* Reads and parses information for one surface energy  specification.
* Current line starts with SURFACE ENERGY keyword.
* Optional GLOBAL keyword.  OBSOLETE and now an error as of version 2.17.
*/

void read_surface_energy()
{
  kb_error(1702,
   "Numbered surface energies are obsolete. Please use named quantity.\n",
      DATAFILE_ERROR);
  tok = yylex();
} // end read_surface_energy()

/*************************************************************************
*
*  function: read_method_instance()
*
*  purpose: reads information for definition of one method instance.
*  return: instance number
*/

int read_method_instance()
{ char mname[40];
  int mi = -1;

  tok = yylex(); /* name */
  if ( tok != NEWIDENT_TOK && tok != METHOD_NAME_TOK )
  { kb_error(1708,"Need instance name.\n",DATAFILE_ERROR); return -1; }

  if ( tok == METHOD_NAME_TOK )
  { mi = yylval.i;  /* redefinition or was forward */
  }
 
  strncpy(mname,yytext,sizeof(mname));
  tok = yylex();
  if ( tok == ';' ) /* forward definition */
  { verb_flag = 0;
    tok = yylex();
    mi = new_method_instance(NULL,mname);
    METH_INSTANCE(mi)->flags |= Q_FORWARD_DEF;
    return mi;
  }
  
  if ( tok != METHOD_TOK )
  { kb_error(1709,"Missing METHOD keyword.\n",DATAFILE_ERROR); return -1; }

  tok = yylex();
  
  if ( tok != NEWIDENT_TOK  && tok != MEAN_CURV_INT_TOK /* kludge */)
  { kb_error(1710,"Need method name.\n",DATAFILE_ERROR); return -1; }
  mi = new_method_instance(yytext,mname);
  
  tok = yylex();
  
  if ( mi >= 0 ) 
     read_instance_attr(mi);
  METH_INSTANCE(mi)->flags &= ~Q_FORWARD_DEF;
  
  return mi;
} // end read_method_instance()

/*************************************************************************
*
* function: read_instance_attr()
*
* purpose: Read attributes of method instance, whether in method_instance
* definition or part of quantity definition.
*/

void read_instance_attr(int mnum  /* number of method instance */)
{
  int bflag;
  int spec_flag = 0;
  int esize,i;
  struct gen_quant_method *gm;
  struct method_instance *mi = METH_INSTANCE(mnum);
  int comps,etype;
  REAL val;
  int expr_count = 0;

  gm = basic_gen_methods + mi->gen_method;
  /* read further attributes of instance */
  for (bflag=0;bflag==0;)
     switch ( tok )
     { case GLOBAL_TOK:
          if ( !(mi->flags & GLOBAL_INST) )
             apply_method_num(NULLID,mnum); 
          tok = yylex(); 
          break;
       case MODULUS_TOK:
          if ( read_const(&(mi->modulus))  <= 0 )
             kb_error(1711,"Missing modulus.\n",WARNING);
          else tok = yylex();
          break;
       case ELEMENT_MODULUS_TOK:
          tok = yylex();
          if ( tok != EXTRA_ATTRIBUTE_TOK )
             kb_error(2123,"Need extra attribute name after ELEMENT_MODULUS.\n",
                DATAFILE_ERROR);
          mi->elmodulus = find_extra(idname,&etype);
          if ( etype != mi->type )
             kb_error(2124,"Extra attribute is for wrong type of element.\n",RECOVERABLE);
          if ( EXTRAS(etype)[mi->elmodulus].type != REAL_TYPE )
             kb_error(2125,"Element modulus attribute type must be REAL.\n",
              DATAFILE_ERROR);
          mi->flags |= ELEMENT_MODULUS_FLAG;
          tok = yylex();
          break;
       case IGNORE_FIXED_TOK:
          mi->flags |= IGNORE_FIXED;
          tok = yylex();
          break;
       case IGNORE_CONSTRAINTS_TOK:
          mi->flags |= IGNORE_CONSTR;
          sqcurve_ignore_constr = 1; /* kludge */
          tok = yylex();
          break;
       case K_VEC_ORDER_TOK:
          if ( read_const(&val)  <= 0 )
          { kb_error(1712,"Missing k_vector_order value.\n",DATAFILE_ERROR);
             val = 1.;
          }
          else tok = yylex();
          mi->vec_order = (int)val;
          break;
       case PARAMETER_1_TOK:
          if ( read_const(&(mi->parameter_1))  <= 0 )
             kb_error(1713,"Missing parameter_1 value.\n",WARNING);
          else tok = yylex();
          mi->flags |= METH_PARAMETER_1;
          break;
       case PARAMETER_2_TOK:
          if ( read_const(&(mi->parameter_2))  <= 0 )
             kb_error(2660,"Missing parameter_2 value.\n",WARNING);
          else tok = yylex();
          mi->flags2 |= METH_PARAMETER_2;
          break;
       case CALC_IN_3D_TOK:
           mi->flags2 |= CALC_IN_3D;
           tok = yylex();
           break;
       case SCALAR_INTEGRAND_TOK: 
           /* read and parse integrand function */
           if ( !(gm->spec_flags & SPEC_SCALAR) )
           { kb_error(1714,"No scalar integrand for this method.\n",
               DATAFILE_ERROR);
             /* read in anyway */
           }
           spec_flag |= SPEC_SCALAR;
           expr_count = 1;
           if ( mi->expr[0] )
           { // in case of replace_load or something
             for ( i = 0 ; i < expr_count ; i++ )
               free_expr(mi->expr[i]);
           }
           mi->expr[0] =
               (struct expnode *)kb_realloc((char*)mi->expr[0],expr_count*sizeof(struct expnode));
           if ( gm->spec_flags & SPEC_EXTRADIM )
           { esize = exparse(2*SDIM,mi->expr[0],USERCOPY);
             spec_flag |= SPEC_EXTRADIM;
           }
           else if ( gm->spec_flags & SPEC_EXTRADIM2 )
           { esize = exparse(SDIM+(SDIM*(SDIM-1))/2,mi->expr[0],USERCOPY);
             spec_flag |= SPEC_EXTRADIM2;
           }
           else esize = exparse(SDIM,mi->expr[0],USERCOPY);
           tok = yylex();
           if ( esize <= 0 )
           { sprintf(errmsg, "Bad integrand definition.\n");
             kb_error(1715,errmsg,DATAFILE_ERROR);
           }
           sprintf(msg, "%s scalar integrand", mi->name);
           strncpy(mi->expr[0]->name,msg,EXPNAMESIZE-1);
           break;

       case VECTOR_INTEGRAND_TOK:
           /* read and parse integrand vector */
           if ( !(gm->spec_flags & (SPEC_VECTOR|SPEC_KVECTOR)) )
           { kb_error(1716,"No vector integrand for this method.\n",
                DATAFILE_ERROR);
             comps = SDIM;
             /* read vector integrand anyway to get past it */
           }
           if ( gm->spec_flags & SPEC_VECTOR )
           { spec_flag |= SPEC_VECTOR;
             comps = SDIM;
           }
           else /* SPEC_KVECTOR  */
           { spec_flag |= SPEC_KVECTOR;
             comps = (SDIM-mi->vec_order)*SDIM;
             if ( comps <= 0 ) 
                 kb_error(1717,"Need k_vector_order.\n",DATAFILE_ERROR);
           }
           if ( comps > MAXMEXPR )
              kb_error(1718,"Total components exceeds MAXMEXPR.\n",DATAFILE_ERROR);
           expr_count = comps;
           if ( mi->expr[0] )
           { // in case of replace_load or something
             for ( i = 0 ; i < expr_count ; i++ )
               free_expr(mi->expr[i]);
           }
           mi->expr[0]=(struct expnode *)kb_realloc((char*)mi->expr[0],
             comps*sizeof(struct expnode));
           tok = yylex();
           for ( i = 0 ; i < comps ; i++ )
           { mi->expr[i] = mi->expr[0] + i;
             if ( (tolower(yytext[0]) != 'q') || (atoi(yytext+1) != 1 + i ))
             { sprintf(errmsg,"Expected component %d definition.\n",i+1);
               kb_error(1719,errmsg,DATAFILE_ERROR);
               break;                  
             }
             esize = exparse(SDIM,mi->expr[i],USERCOPY);
             tok = yylex();
             if ( esize <= 0 )
             { sprintf(errmsg, "Bad component %d definition.\n",i+1);
               kb_error(1720,errmsg,DATAFILE_ERROR);
             }
             sprintf(msg,"%s vector integrand component %d",
                mi->name,i+1);
             strncpy(mi->expr[i]->name,msg,EXPNAMESIZE-1);
           }
           if ( (tolower(yytext[0]) == 'q') && isdigit(yytext[1]) ) 
           { sprintf(errmsg,"k_vector method needs only %d (complement of order) vectors.\n",(SDIM-METH_INSTANCE(mnum)->vec_order));
             kb_error(2863,errmsg,RECOVERABLE);
           }
           break;

       case FORM_INTEGRAND_TOK:  /* 2-form */
           /* read and parse integrand 2-form */
           if ( !(gm->spec_flags & SPEC_2FORM) )
           { kb_error(1721,"No 2-form integrand for this method.\n",
                   DATAFILE_ERROR);
             /* read in anyway to get past it */
           }
           spec_flag |= SPEC_2FORM;
           comps = (SDIM*(SDIM-1))/2;
           if ( comps > MAXMEXPR )
              kb_error(1722,"Total components exceeds MAXMEXPR.\n",
                 DATAFILE_ERROR);

           expr_count = comps;
           if ( mi->expr[0] )
           { // in case of replace_load or something
             for ( i = 0 ; i < expr_count ; i++ )
               free_expr(mi->expr[i]);
           }

           mi->expr[0]=(struct expnode *)kb_realloc((char*)mi->expr[0],
             comps*sizeof(struct expnode));
           tok = yylex();
           for ( i = 0 ; i < comps ; i++ )
           {
             mi->expr[i] = mi->expr[0] + i;
             if ( (tolower(yytext[0]) != 'q') || (atoi(yytext+1) != 1 + i ) )
             { sprintf(errmsg,"Expected component %d definition.\n",i+1);
               kb_error(1723,errmsg,DATAFILE_ERROR);
             }
             esize = exparse(SDIM,mi->expr[i],USERCOPY);
             tok = yylex();
             if ( esize <= 0 )
             { sprintf(errmsg, "Bad component %d definition.\n",i+1);
               kb_error(1724,errmsg,DATAFILE_ERROR);
             }
             sprintf(msg,"%s form integrand component %d", mi->name,i+1);
             strncpy(mi->expr[i]->name,msg,EXPNAMESIZE-1);
           }
           break;

       default: bflag = 1; break;
     }
    if ( (gm->spec_flags & SPEC_SCALAR) & ~spec_flag )
      kb_error(1725,"Need scalar integrand for this method.\n",DATAFILE_ERROR);

    if ( (gm->spec_flags & SPEC_VECTOR) & ~spec_flag )
      kb_error(1726,"Need vector integrand for this method.\n",DATAFILE_ERROR);

    if ( (gm->spec_flags & SPEC_KVECTOR) & ~spec_flag )
      kb_error(2862,
        "Need k_vector_order and vector_integrand for this method.\n",
           DATAFILE_ERROR);

    /* Check use of boundary parameters */
    for ( i = 0 ; i < expr_count ; i++ )
	{ 
	  if ( mi->expr[i] && (mi->expr[i]->flag & USING_PARAM_FLAG) )
      { sprintf(errmsg,
         "%s: Cannot use boundary parameters in gradient or hessian.\n",
           METH_INSTANCE(mnum)->name);
	    kb_error(3895,errmsg,WARNING);
	  }
    }
    
    mi->flags &= ~Q_FORWARD_DEF;
 } // end read_instance_attr()

/*************************************************************************
*
* Reads and parses information for one quantity specification, 
* in particular, facet integrands and fixed value.
* Current line starts with QUANTITY keyword.
* Optional GLOBAL keyword.
*/

int read_quantity()
{
  tok = gettok(INTEGER_TOK);
  if ( tok != INTEGER_TOK ) 
  { return read_named_quantity();
  }
  kb_error(1727,
   "Numbered quantities are obsolete. Please use named quantity.\n",
      DATAFILE_ERROR);
  tok = yylex();
  return 0;
} // end read_quantity()

/***********************************************************************
*
*  function: read_named_quantity()
*
*  purpose: parse named quantity definition.
*  return: quantity number.
*/

int read_named_quantity()
{
  int n;
  int gnum; /* quantity number */
  char qname[400];
  int meth;
  struct method_instance *mi;
  char inst_name[400];
  int namecount; /* number of uninstantiated methods */
  int globality;
  int esize;
  int lagmulflag = 0;

  if ( (tok != NEWIDENT_TOK) && (tok != IDENT_TOK) && (tok != QUANTITY_NAME_TOK) )
  { kb_error(1733,"Need quantity name.\n",DATAFILE_ERROR);
    return -1;
  }
  strncpy(qname,yytext,sizeof(qname));
  /* check not already defined and add to list */
  tok = yylex();
  if ( tok == ';' ) 
  { gnum = new_quantity(qname,Q_INFO);
    GEN_QUANT(gnum)->flags |= Q_FORWARD_DEF;
    verb_flag = 0; /* set by ';' */
    tok = yylex();
    return gnum;
  }
  if ( tok == FIXED_TOK )
  { gnum = new_quantity(qname,Q_FIXED);
    tok = yylex(); /* eat '=' */
    if ( tok != '=' )
      kb_error(1734,"Missing '='\n",DATAFILE_ERROR);
    if (read_const(&(GEN_QUANT(gnum)->target)) <= 0)
      kb_error(1735,"Missing quantity target value. \n",DATAFILE_ERROR);
    tok = yylex();
  }
  else if ( tok == ENERGY_TOK )
  { gnum = new_quantity(qname,Q_ENERGY);
    tok = yylex();
  }
  else if ( tok == INFO_ONLY_TOK )
  { gnum = new_quantity(qname,Q_INFO);
    tok = yylex();
  }
  else if ( tok == CONSERVED_TOK )
  { gnum = new_quantity(qname,Q_CONSERVED);
    tok = yylex();
  }
  else 
  { kb_error(1736,
      "Need type of quantity: energy, fixed, conserved, or info_only.\n",
           DATAFILE_ERROR);
    gnum = new_quantity(qname,Q_INFO);
  }
  namecount = 0;
  for (;;) /* further attributes */
    switch ( tok )
    { case GLOBAL_METHOD_TOK :
      case METHOD_TOK :
          if ( GEN_QUANT(gnum)->flags & Q_COMPOUND )
            kb_error(1737,
              "Can't list separate methods with function of methods.\n",
                         DATAFILE_ERROR);
          globality = tok;
          tok = yylex();

          strncpy(inst_name,qname,sizeof(qname));
          sprintf(inst_name+strlen(inst_name),"%d_",++namecount);
          strncat(inst_name,yytext,sizeof(qname)-strlen(inst_name));

          /* see if existing instance */
          for ( n=LOW_INST ; n < meth_inst_count ; n++ )
          { mi = METH_INSTANCE(n);
            if ( mi->flags & Q_DELETED ) continue;
            if ( stricmp(mi->name,inst_name) == 0 ) 
            { // existing implicit method, say in replace_load
              attach_method_num(gnum,n);
              if ( globality == GLOBAL_METHOD_TOK) 
                apply_method(NULLID,inst_name);
              tok = yylex();
              read_instance_attr(n);
              break;
            }
            if ( stricmp(mi->name,yytext) == 0 )
            { /* predefined instance */
              int mnum = attach_method(gnum,yytext);
              if ( (globality  == GLOBAL_METHOD_TOK) && (mnum>=0) &&
                !(METH_INSTANCE(mnum)->flags & GLOBAL_INST) )
                 apply_method(NULLID,yytext);
              tok = yylex();
              break;
            }
          }
          if ( n >= meth_inst_count )
          { /* need to instantiate method */
            meth = new_method_instance(yytext,inst_name);
            if ( meth >= 0 )
            { METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE;
              attach_method_num(gnum,meth);
              if ( globality == GLOBAL_METHOD_TOK) 
                apply_method(NULLID,inst_name);
              tok = yylex();
              read_instance_attr(meth);
            }
          }
          break;

      case LAGRANGE_MULTIPLIER_TOK: 
         if ( read_const(&(GEN_QUANT(gnum)->pressure)) <= 0 ) 
           kb_error(2126,"Missing lagrange_multiplier value.\n",DATAFILE_ERROR);  
         else 
         { GEN_QUANT(gnum)->flags |= Q_PRESSURE_SET; 
           lagmulflag = 1; tok = yylex(); 
         }
         break;

      case VOLCONST_TOK:
          if (read_const(&(GEN_QUANT(gnum)->volconst)) <= 0)
             kb_error(1738,"Missing quantity volconst value. \n",DATAFILE_ERROR);
          else tok = yylex();
          break;
      case TOLERANCE_TOK:
         if ( read_const(&GEN_QUANT(gnum)->tolerance) <= 0 )
            kb_error(2127,"Missing tolerance value.\n",DATAFILE_ERROR);
         else tok=yylex();
         if ( GEN_QUANT(gnum)->tolerance <= 0.0 ) 
          kb_error(2128,"Tolerance must be positive.\n",DATAFILE_ERROR);
         break;

      case MODULUS_TOK:
          if (read_const(&(GEN_QUANT(gnum)->modulus)) <= 0)
             kb_error(1739,"Missing quantity modulus value. \n",DATAFILE_ERROR);
          else tok = yylex();
          break;
      case FUNCTION_TOK:  /* compound function of instances */
          cur_quant = gnum;
          if ( GEN_QUANT(cur_quant)->method_count > 0 && !addload_flag)
            kb_error(1740,
              "Can't list separate methods with function of methods.\n",
                         DATAFILE_ERROR);
          reading_comp_quant_flag = 1;
          esize = exparse(0,&(GEN_QUANT(gnum)->expr),USERCOPY);
          reading_comp_quant_flag = 0;
          cur_quant = -1 ;
          tok = yylex();
          if ( esize <= 0 )
          { sprintf(errmsg, "Bad function definition.\n");
            kb_error(1741,errmsg,DATAFILE_ERROR);
          }
          sprintf(msg,"%s formula",GEN_QUANT(gnum)->name);
          strncpy(GEN_QUANT(gnum)->expr.name,msg,EXPNAMESIZE-1);
          GEN_QUANT(gnum)->flags |= Q_COMPOUND;
          if ( gnum > compound_quant_list_head ) // avoid loop with addload
          { GEN_QUANT(gnum)->next_compound = compound_quant_list_head;
            compound_quant_list_head = gnum;
          }
          break;
      default:
          goto named_exit; /* done with quantity */
          
  }

named_exit:
  if ( !lagmulflag ) pressure_set_flag = 0;
  GEN_QUANT(gnum)->flags &= ~Q_FORWARD_DEF;
  return gnum;
} // end read_named_quantity()

/*************************************************************
*
*  Function: add_outside()
*
*  Purpose:  Adds body outside all other bodies.  Used in
*                dynamic pressure calculations.
*/

void add_outside()
{

return; /* temporary turn off */
#ifdef OUTSIDEBODY     
    body_id b_id;    /* both models */


        b_id = new_body();
        web.outside_body = b_id;
        set_attr(b_id,PRESSURE); /* since has ambient pressure */

        if ( web.representation == STRING )
         { edge_id e_id;
            facet_id f_id;

            f_id = new_facet();    /* outside facet */
            set_facet_body(f_id,b_id);

            /* add to outside of every edge bordering just ONE cell */
            FOR_ALL_EDGES(e_id)
              {
                 facetedge_id new_fe;
                 facetedge_id fe_id;
                 
                 fe_id = get_edge_fe(e_id);
                 if ( !valid_id(fe_id) ) continue;
                 if ( !equal_id(fe_id,get_next_facet(fe_id)) ) continue;
                 new_fe = new_facetedge(inverse_id(f_id),e_id);
                 set_next_facet(fe_id,new_fe);
                 set_prev_facet(fe_id,new_fe);
                 set_next_facet(new_fe,fe_id);
                 set_prev_facet(new_fe,fe_id);
                 set_facet_fe(f_id,new_fe);

              }
         }
      else /* SOAPFILM */
         {
            facet_id f_id;

            /* add to outside of every facet bordering just ONE cell */
            FOR_ALL_FACETS(f_id)
              {
                 body_id b1_id,b2_id;

                 b1_id = get_facet_body(f_id);
                 b2_id = get_facet_body(inverse_id(f_id));
                 if ( valid_id(b1_id) == valid_id(b2_id) ) continue;

                 if ( valid_id(b1_id) )
                    set_facet_body(inverse_id(f_id),b_id);
                 if ( valid_id(b2_id) )
                    set_facet_body(f_id,b_id);
              }
        }
#endif
} // end  add_outside()

/*********************************************************************
*
*  Function: fix_volconst()
*
*  Purpose:  For torus, figure out constant volume adjustments, 
*                figuring that given volumes are within 1/2 of a 
*                torus volume of their true value
*/

void fix_volconst()
{
  REAL adjust;
  body_id b_id;

  if ( web.pressure_flag ) return;

  /* get volume of piece of unit cell */
  if ( SDIM == 2 )
  {
     adjust = web.torusv  /* /2 */;
  }
  else /* web.representation == SOAPFILM */
  {
     adjust = web.torusv  /* /6 */;
  }

  /* adjust volconsts */
  FOR_ALL_BODIES(b_id)
    if ( get_battr(b_id) & FIXEDVOL )
    { REAL vol = get_body_volume(b_id);
      REAL fix = get_body_fixvol(b_id);
      REAL vc = get_body_volconst(b_id);
      REAL calcvol = vol-vc;
      REAL newvc = fix - calcvol;
      newvc = adjust*floor(0.5+newvc/adjust);
      set_body_volconst(b_id,newvc);
      set_body_volume(b_id,calcvol+newvc,SETSTAMP);
   }
} // end fix_volconst()

/*******************************************************************
*
*  Function: fe_reorder()
*
*  Purpose:  Order facets properly around edge.
*  NOTE: Does have body agreement override geometry, so be careful
*        about body setup before entry.
*/

struct fsort { facetedge_id fe;
                    REAL angle;
             };

static int fcompare(a,b)
struct fsort *a,*b;
{
  if ( a->angle < b->angle ) return -1;
  if ( a->angle > b->angle ) return 1;
  return 0;
} // end fcompare()

void fe_reorder(edge_id e_id)
{ int fcount = 0;
  facetedge_id fe;
#define FSORTMAX 100
  struct fsort fe_list[FSORTMAX];
  REAL side[MAXCOORD],norm_a[MAXCOORD],side_a[MAXCOORD];
  REAL side_b[MAXCOORD],norm_aa[MAXCOORD],a_norm,aa_norm;
  REAL c,s,angle;
  int i,j,k;
  facetedge_id first_fe;
  int bodies_ok;
  int body_order = 0; /* set if order triple line by bodies */

  /* see if we have 3 or more facets */
  fe = first_fe = get_edge_fe(e_id);
  if ( valid_id(fe) ) 
  do { fcount++; 
       fe = get_next_facet(fe);
     } while ( valid_id(fe) && !equal_id(fe,first_fe) );
  if ( fcount <= 2 ) return;

  if ( fcount > FSORTMAX )
  { kb_error(1742,"More than 100 facets on an edge; not sorted.\n",WARNING); 
    return;
  }

  /* common case of three facets, check for body determination */
  if ( fcount == 3 )
  { /* see if there is a unique order of facets */
    facetedge_id fa = get_next_facet(fe);
    facetedge_id fb = get_prev_facet(fe);
    body_id ef = get_facet_body(get_fe_facet(fe));
    body_id eb = get_facet_body(inverse_id(get_fe_facet(fe)));
    body_id af = get_facet_body(get_fe_facet(fa));
    body_id ab = get_facet_body(inverse_id(get_fe_facet(fa)));
    body_id bf = get_facet_body(get_fe_facet(fb));
    body_id bb = get_facet_body(inverse_id(get_fe_facet(fb)));
    int asis  = (eb==af) && (ab==bf) && (bb==ef);
    int other = (eb==bf) && (bb==af) && (ab==ef);
    if ( asis && !other ) body_order = 1;
    else if ( other && !asis )
    { /* swap */
      set_next_facet(fe,fb);
      set_next_facet(fb,fa);
      set_next_facet(fa,fe);
      set_prev_facet(fe,fa);
      set_prev_facet(fa,fb);
      set_prev_facet(fb,fe);
      body_order = 1;
    }
    /* else fall through to geometric test */
  }

  /* use first facet as reference facet */
  /* to get basis for calculating angles */
  get_edge_side(e_id,side);     
  fe = get_edge_fe(e_id);
  fe_list[0].fe = fe;
  get_fe_side(get_next_edge(fe),side_a);
  cross_prod(side,side_a,norm_a);
  cross_prod(norm_a,side,norm_aa);
  a_norm = sqrt(SDIM_dot(norm_a,norm_a));
  aa_norm = sqrt(SDIM_dot(norm_aa,norm_aa));

  /* now get angles to rest of facets */
  fe_list[0].angle = 0.0;
  for ( i = 1 ; i < fcount ; i++ )
  { fe = get_next_facet(fe);
    fe_list[i].fe = fe;
    get_fe_side(get_next_edge(fe),side_b);
    s = SDIM_dot(side_b,norm_a)*aa_norm;
    c = SDIM_dot(side_b,norm_aa)*a_norm;
    angle = atan2(s,c);
    fe_list[i].angle = angle;
  }
  if ( body_order )
  { /* warn if not agreeing with geometric order */
    if ( ((fe_list[1].angle > 0) && (fe_list[2].angle > fe_list[0].angle) && 
           (fe_list[2].angle < fe_list[1].angle))  ||
      (( (fe_list[1].angle < 0) && ((fe_list[2].angle > fe_list[0].angle) || 
           (fe_list[2].angle < fe_list[1].angle)))   )
      )
    { sprintf(errmsg,
         "Edge %s facets body order disagrees with geometric order.\n",
            ELNAME(e_id));
      kb_error(3857,errmsg,WARNING);
    } 
    return;
  }

  /* sort by angle */
/* Bombed in IRIX 6 long double mode
  qsort((char *)&fe_list[1],fcount-1,sizeof(struct fsort),FCAST fcompare);
*/
  for ( j = 0 ; j < fcount-1 ; j++ )
    for ( k = j+1 ; k < fcount ; k++ )
       if ( fe_list[k].angle < fe_list[j].angle )
       { struct fsort ftemp;
         ftemp = fe_list[k];
         fe_list[k] = fe_list[j];
         fe_list[j] = ftemp;
       } 

  /* check consistency for facets on bodies */
  bodies_ok = 1;
  for ( i = 0 ; i < fcount ; i++ )
  { facet_id f1,f2;
    int ii = (i == fcount-1) ? 0 : i+1;
    f1 = facet_inverse(get_fe_facet(fe_list[i].fe));
    f2 = get_fe_facet(fe_list[ii].fe);
    if ( !equal_id(get_facet_body(f1),get_facet_body(f2)) )
    { bodies_ok = 0;
      break;
    }
  }

  if ( !bodies_ok )
  { // see if body matching forces order
    bodies_ok = 1;
    for ( k = 0 ; k < fcount ; k++ )
    { int kk;
      for ( kk = k+1 ; kk < fcount+k ; kk++ )
      { int kkk = (kk >= fcount) ? kk-fcount : kk;
        body_id kbod,kkbod;
        kbod = get_facet_body(inverse_id(get_fe_facet(fe_list[k].fe)));
        kkbod = get_facet_body(get_fe_facet(fe_list[kkk].fe));
        if ( kbod == kkbod )
        { int kkkk;
          struct fsort tempfe;
          if ( kk == k+1 )
            break;
          // swap 
          kkkk = (k+1 >= fcount) ? k+1-fcount : k+1;
          tempfe = fe_list[kkkk];
          fe_list[kkkk] = fe_list[kkk];
          fe_list[kkk] = tempfe;
          break;
        }
      }
      if ( kk == fcount+k )
        bodies_ok = 0;
    }
  }

  if ( !bodies_ok )
  { REAL best_violation;
    int k,kk,bestk;
    /* try swaps of adjacent facets, and pick workable one with
       least violation of angle ordering */
    best_violation = 1e30;
    bestk = -1;
    for ( k = 0 ; k < fcount ; k++ )
    { struct fsort ftemp;
      kk = (k==fcount-1) ? 0 : k+1;
      ftemp = fe_list[k];
      fe_list[k] = fe_list[kk];
      fe_list[kk] = ftemp;
 
      bodies_ok = 1;
      for ( i = 0 ; i < fcount-1 ; i++ )
      { facet_id f1,f2;

        f1 = facet_inverse(get_fe_facet(fe_list[i].fe));
        f2 = get_fe_facet(fe_list[i+1].fe);
        if ( !equal_id(get_facet_body(f1),get_facet_body(f2)) )
        { bodies_ok = 0;
          break;
        }
      }
      if ( bodies_ok )
      { REAL violation = fabs(fe_list[k].angle-fe_list[kk].angle);
        if ( violation > M_PI ) 
           violation = 2*M_PI - violation;  /* shortest way */
        if ( violation < best_violation )
        { bestk = k;
          best_violation = violation;
        }
      }
      ftemp = fe_list[kk];
      fe_list[kk] = fe_list[k];
      fe_list[k] = ftemp;
    }
    if ( bestk >= 0 )
    { struct fsort ftemp;
      kk = (bestk==fcount-1) ? 0 : bestk+1;
      ftemp = fe_list[bestk];
      fe_list[bestk] = fe_list[kk];
      fe_list[kk] = ftemp;
    }
  }

  /* relink in proper order */
  for ( i = 0 ; i < fcount ; i++ )
  { set_next_facet(fe_list[i].fe,fe_list[(i+1)%fcount].fe);
    set_prev_facet(fe_list[i].fe,fe_list[(i+fcount-1)%fcount].fe);
  }
} // end fe_reorder()

/*******************************************************************
*
*  Function: raw_fe_reorder()
*
*  Purpose:  Order facets properly around edge, without regard for bodies.
*            For use by merge_edge().
*            Note: for higher dimension ambient space, just uses 
*            first three dimensions.
*/
void raw_fe_reorder(edge_id e_id)
{ int fcount = 0;
  facetedge_id fe;
  struct fsort fe_list[FSORTMAX];
  REAL side[MAXCOORD],norm_a[MAXCOORD],side_a[MAXCOORD];
  REAL side_b[MAXCOORD],norm_aa[MAXCOORD],a_norm,aa_norm;
  REAL c,s,angle;
  int i,j,k;
  facetedge_id first_fe;

  memset(norm_a,0,sizeof(norm_a));
  memset(norm_aa,0,sizeof(norm_aa));

  /* see if we have 3 or more facets */
  fe = first_fe = get_edge_fe(e_id);
  if ( valid_id(fe) ) 
  do { fcount++; 
       fe = get_next_facet(fe);
     } while ( valid_id(fe) && !equal_id(fe,first_fe) );
  if ( fcount <= 2 ) return;

  if ( fcount > FSORTMAX )
  { kb_error(5742,"More than 100 facets on an edge; not sorted.\n",WARNING); 
    return;
  }

  /* use first facet as reference facet */
  /* to get basis for calculating angles */
  get_edge_side(e_id,side);     
  fe = get_edge_fe(e_id);
  fe_list[0].fe = fe;
  get_fe_side(get_next_edge(fe),side_a);
  cross_prod(side,side_a,norm_a);
  cross_prod(norm_a,side,norm_aa);
  a_norm = sqrt(SDIM_dot(norm_a,norm_a));
  aa_norm = sqrt(SDIM_dot(norm_aa,norm_aa));

  /* now get angles to rest of facets */
  fe_list[0].angle = 0.0;
  for ( i = 1 ; i < fcount ; i++ )
  { fe = get_next_facet(fe);
    fe_list[i].fe = fe;
    get_fe_side(get_next_edge(fe),side_b);
    s = SDIM_dot(side_b,norm_a)*aa_norm;
    c = SDIM_dot(side_b,norm_aa)*a_norm;
    angle = atan2(s,c);
    fe_list[i].angle = angle;
  }

  /* sort by angle */
  for ( j = 0 ; j < fcount-1 ; j++ )
    for ( k = j+1 ; k < fcount ; k++ )
       if ( fe_list[k].angle < fe_list[j].angle )
       { struct fsort ftemp;
         ftemp = fe_list[k];
         fe_list[k] = fe_list[j];
         fe_list[j] = ftemp;
       } 

  /* relink in proper order */
  for ( i = 0 ; i < fcount ; i++ )
  { set_next_facet(fe_list[i].fe,fe_list[(i+1)%fcount].fe);
    set_prev_facet(fe_list[i].fe,fe_list[(i+fcount-1)%fcount].fe);
  }
} // end raw_fe_reorder()


/**************************************************************
*
*  Function: gettok()
*
*  Purpose:  get next integer or real value, possibly
*                with '-' preceding.
*/

int gettok(int kind /* REAL_TOK or INTEGER_TOK */ )
{
  int sign = 1; 

  tok = yylex();
  if ( tok == ',' ) tok = yylex(); /* skip separating comma */
  if ( tok == '-' ) { sign = -1; tok = yylex(); }
  if ( tok == UMINUS_TOK ) { sign = -1; tok = yylex(); }
  if ( tok != kind )
  { if ( !((tok == INTEGER_TOK) && (kind == REAL_TOK) ) &&
         !((tok == INTEGER_AT_TOK) && (kind == INTEGER_TOK)) )
    { if ( sign == -1 ) 
          kb_error(2129,"Unexpected minus sign.\n",DATAFILE_ERROR);
      return tok;
    }
    /* caller should check for error, and if error leave tok as lookahead */
  }

  yylval.i *= sign;
  yylval.r *= sign;
  tok = kind;
  return tok;
} // end gettok()


/*******************************************************************
*
*  function: read_const()
* 
*  purpose: read constant expression from datafile
*              and return value.
*
*  input:    REAL *value  - address for value
*
*  return:  < 0 error
*             0 no expression
*           > 0 valid expression
*/

int read_const(REAL *value)
{
  int retval;
  struct expnode node;  /* for getting constant expression */

  memset(&node,0,sizeof(struct expnode));
  const_expr_flag = 1;
  if ( (retval = exparse(0,&node,NOUSERCOPY)) <= 0 )
  { const_expr_flag = 0;
    return retval;
  }
  /* sprintf(node.name,"constant expression at line %d",line_no); */
  *value = eval(&node,NULL,NULLID,NULL);
  if ( node.locals ) 
  { if ( node.locals->list ) myfree((char*)node.locals->list );
    myfree((char*)(node.locals));
  }
  const_expr_flag = 0;
  return 1;
} // end read_const()

/*********************************************************************
*
* function: const_expr()
*
* purpose: get numerical value of string expresssion
*
*  input:    char *str - string with expression
*              REAL *value  - address for value
*
*  return:  < 0 error
*                 0 no expression
*              > 0 valid expression
*/

int const_expr(
  char *str,
  REAL *value
)
{ char *old_cmdptr;  /* in case in middle of another parse */
  int ret;    /* return value */

  if ( str == NULL ) return 0;
  old_cmdptr = cmdptr;
  cmdptr = str;
  yylex_init();
  ret = read_const(value);
  cmdptr = old_cmdptr;
  return ret;
} // end const_expr()


/*********************************************************************
*
*  function: string_fixup()
*
*  purpose: put minimal facet-edges on string net, so can
*           do edge changes.  all facet-edges given the null facet.
*
*/

void string_fixup()
{
  edge_id e_id;
  facetedge_id fe;

  /* do all bare edges */
  FOR_ALL_EDGES(e_id)
    {
      fe = get_edge_fe(e_id);
      if ( valid_id(fe) ) continue;

      fe = new_facetedge(NULLFACET,e_id);
      set_next_facet(fe,fe);
      set_prev_facet(fe,fe);
      set_next_edge(fe,inverse_id(fe));
      set_prev_edge(fe,inverse_id(fe));
      set_edge_fe(e_id,fe);

    }
} // end string_fixup()


/*******************************************************
*
*  phase_initialize()
*
*  Purpose: Read in phase boundary energies from file  
*
*  Input:    Name of file with phase boundary energies
*              First line has number of phases
*              Succeeding lines have pair of phase numbers
*                      and boundary energy.
*
*/

void phase_initialize(char *phasename)
{
  FILE *pfd;
  int i,j;         /* which phases */
  REAL value;
  char line[200];
  int one = 1;

  phase_flag = 1;

  if ( web.representation == STRING )
    F_PHASE_ATTR = add_attribute(FACET,"phase",INTEGER_TYPE,0,&one,0,NULL,MPI_NO_PROPAGATE);
  else
    B_PHASE_ATTR = add_attribute(BODY,"phase",INTEGER_TYPE,0,&one,0,NULL,MPI_NO_PROPAGATE);

  /* save  name */
  strncpy(phase_file_name,phasename,sizeof(phase_file_name));

  pfd = path_open(phasename,NOTDATAFILENAME);
  if ( pfd == NULL )
  { sprintf(errmsg, "Cannot open phase boundary energy file %s.\n",phasename);
    kb_error(1746,errmsg, DATAFILE_ERROR);
    return;
  }

  fscanf(pfd,"%d",&phasemax);
  phase_data = dmatrix(0,phasemax,0,phasemax);
  for ( i = 0 ; i <= phasemax ; i++ )
     for ( j = 0 ; j <= phasemax ; j++ ) 
        phase_data[i][j] = 1.0;
  while ( fgets(line,sizeof(line),pfd) )
  { 
#ifdef FLOAT128
    char *c;
    c = strtok(line," \t");
    if ( !c ) continue;
    i = atoi(c);
    c = strtok(NULL," \t");
    if ( !c ) continue;
    j = atoi(c);
    c = strtok(NULL," \t");
    if ( !c ) continue;
    value = atof(c);
#elif defined(LONGDOUBLE)
    if ( sscanf(line,"%d %d %Lf",&i,&j,&value) == 3 ) 
#else
    if ( sscanf(line,"%d %d %lf",&i,&j,&value) == 3 ) 
#endif
    { if ( (i < 0) || (i > phasemax) || (j < 0) || (j > phasemax) )
      { sprintf(errmsg,"Bad phase numbers: %d %d\n",i,j);
        kb_error(1747,errmsg,DATAFILE_ERROR);
      }
      phase_data[i][j] = phase_data[j][i] = value;
    }
  }
  fclose(pfd);
} // end phase_initialize()

/****************************************************************
*
*  Function: read_transforms()
*
*  Purpose:  Reads additional view transforms.
*                Works both from datafile and commandline
*/

void read_transforms(int count /* number, if known from command */ )
{
  int i,j,n;
  REAL value;
  MAT2D(temp_mat,MAXCOORD+1,MAXCOORD+1);

  lists_flag = 1;
  if ( count > 0 ) transform_count = count+1;
  else
  {
    /* find how many transforms */
    if ( (tok = gettok(INTEGER_TOK)) != INTEGER_TOK )
    { kb_error(1748,"Missing number of transforms.\n",DATAFILE_ERROR);
      transform_count = 1;
    }
    else transform_count = yylval.i+1;     /* including identity */
  }

  if ( view_transforms )   
  { free_matrix3(view_transforms); view_transforms = NULL;}

  if ( transform_count <= 0 ) return;
  view_transforms = dmatrix3(transform_count,SDIM+1,SDIM+1);
  set_view_transforms_global();
  allocate_transform_colors(transform_count);
  view_transform_det = (int*)mycalloc(transform_count,sizeof(int));
  matcopy(view_transforms[0],identmat,SDIM+1,SDIM+1);
  transform_colors[0] = SAME_COLOR;
  view_transform_det[0] = 1;

  /* read in transform matrices, in homogeneous coords */
  for ( n = 1 ; n < transform_count ; n++ )
  { tok = yylex();
    if ( tok == COLOR_TOK )
    { if ( (tok = gettok(INTEGER_TOK)) != INTEGER_TOK )
         kb_error(1749,"Missing transform color.\n",DATAFILE_ERROR);
      transform_colors[n] = yylval.i;
      transform_colors_flag = 1;
    }
    else if ( tok == SWAP_COLORS_TOK )
    { transform_colors[n] = SWAP_COLORS;
      transform_colors_flag = 1;
    }
    else
    { transform_colors[n] = SAME_COLOR;
      unput_tok();
    }
    for ( i = 0 ; i <= SDIM ; i++ )
    { for ( j = 0 ; j <= SDIM ; j++ ) 
      { if ( read_const(&value) <= 0 )
        { kb_error(1750,"Not enough values for transform matrix.\n",
             DATAFILE_ERROR);
          return;
        }
        view_transforms[n][i][j] = value;
      }     
      matcopy(temp_mat,view_transforms[n],SDIM+1,SDIM+1);
      if ( determinant(temp_mat,SDIM+1) > 0.0 )
         view_transform_det[n] = 1;
      else  view_transform_det[n] = -1;
    }
  }
  transforms_flag = 1; /* default is to show */
  lists_flag = 0;
  if ( n == transform_count ) tok = yylex(); /* lookahead */
} // end read_transforms()

/****************************************************************
*
* Function: convert_constraint_to_quantities()
*
* Purpose: Create method for constraint energy.
*/
void convert_constraint_to_quantities( int i /* number of constraint */)    
{   int j;
    char qname[100];
    char inst_name[100];
    int gq;
    int meth;
    edge_id e_id;
    struct constraint *con = get_constraint(i);
    if ( !(con->attr & CON_ENERGY) ) return;
    if ( con->attr & NAMED_THING )
    { sprintf(qname,"constraint_%s_energy",con->name);
      sprintf(inst_name,"constraint_%s_energy_inst",con->name);
    }
    else
    { sprintf(qname,"constraint_%d_energy",i);
      sprintf(inst_name,"constraint_%d_energy_inst",i);
    }
    gq = new_quantity(qname,Q_ENERGY);
    GEN_QUANT(gq)->flags |= DEFAULT_QUANTITY;
    if ( web.representation ==  STRING )
       meth = new_method_instance("vertex_scalar_integral",inst_name);
    else
       meth = new_method_instance("edge_vector_integral",inst_name);
    METH_INSTANCE(meth)->connum = i;
    METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE|DEFAULT_INSTANCE;
    attach_method_num(gq,meth);
    if ( con->attr & GLOBAL ) 
       apply_method_num(NULLID,meth);
    else
    { vertex_id v_id;
      if ( web.representation == STRING )
      { FOR_ALL_VERTICES(v_id)
          if ( v_on_constraint(v_id,i) ) 
            apply_quantity(v_id,gq);
      }
      else 
      { FOR_ALL_EDGES(e_id)
          if ( e_on_constraint(e_id,i) ) 
             apply_quantity(e_id,gq);
      }
    }
    for ( j = 0 ; j < SDIM ; j++ ) 
       METH_INSTANCE(meth)->expr[j] = con->envect[j];
    con->attr |= USURPED_BY_QUANTITY;  /* prevent dual expr deallocation */
    con->energy_method = meth;

} // end convert_constraint_to_quantities()
  
/****************************************************************
*
*  Function: read_transform_generators()
*
*  Purpose:  Reads view transform generators.
*                Works both from datafile and commandline
*/

void read_transform_generators(int count /* number, if known from command */ )
{
  int i,j,n;

  // take care of addload memory leak
  if (  view_transform_gens_expr )
  { for ( i = 0; i < transform_gen_count ; i++ )
      for ( j = 0 ; j <= SDIM ; j++ )
        for ( n = 0 ; n <= SDIM ; n++ )
          free_expr(&view_transform_gens_expr[i][j][n]);
    myfree((char*)view_transform_gens_expr);
    view_transform_gens_expr = NULL;
    transform_gen_count = 0;
  }
  if ( view_transform_gens )
    free_matrix3(view_transform_gens);

  lists_flag = 1;
  if ( count > 0 ) transform_count = count;
  else
  { /* find how many transforms */
    if ( (tok = gettok(INTEGER_TOK)) != INTEGER_TOK )
    { kb_error(1751,"Missing number of view_transform_generators.\n",DATAFILE_ERROR);
      transform_gen_count = 1;
    }
    else transform_gen_count = yylval.i;
  }
  if ( transform_gen_count < 0 )
  { kb_error(3900,"Missing number of view_transform_generators.\n",DATAFILE_ERROR);
    transform_gen_count = 0;
  }
 

  if ( transform_gen_swap ) myfree((char*)transform_gen_swap);
  transform_gen_swap = (int*)mycalloc(transform_gen_count+SDIM,sizeof(int));
  view_transform_gens = dmatrix3(web.torus_flag?(transform_gen_count+SDIM):
            transform_gen_count,SDIM+1,SDIM+1);
  view_transform_gens_expr = 
        (expnodearray *)mycalloc((MAXCOORD+1)*(MAXCOORD+1)*(transform_gen_count+SDIM),
                                    sizeof(struct expnode));

  /* read in transform matrices, in homogeneous coords */
  for ( n = 0 ; n < transform_gen_count ; n++ )
  { tok = yylex();
    if ( tok == SWAP_COLORS_TOK )
    { transform_gen_swap[n] = 1;
    }
    else unput_tok();
    for ( i = 0 ; i <= SDIM ; i++ )
    { for ( j = 0 ; j <= SDIM ; j++ ) 
      { int esize = exparse(0,&view_transform_gens_expr[n][i][j],USERCOPY);
        if ( esize < 0 )
        { kb_error(1752,"Not enough entries for transform generator matrix.\n",
              DATAFILE_ERROR);
          return;
        }
      }     
    }
  }
  if ( n == transform_gen_count ) tok = yylex(); /* lookahead */
  if ( web.torus_flag ) transform_gen_count += SDIM;

  transforms_flag = 1; /* default is to show */
  lists_flag = 0;
  calc_view_transform_gens();
} // end read_transform_generators()

/***************************************************************************
*
* function: convert_to_quantities()
*
* purpose: convert numbered quantities, surface energies, and 
*             constraint integrals to named quantities. 
*             Used in everything_quantities mode.
*/

void convert_to_quantities()
{
  int gq;
  int meth;
  char inst_name[40];
  char qname[100];
  int i,j;
  body_id b_id;
  edge_id e_id; 
  int q;
  char formula[1000];
  char *gformula;

  if ( everything_quantities_flag ) return;

  if ( web.wulff_flag && (web.dimension != 2) ) 
     kb_error(1754,"Can't do wulff energy method for strings yet.\n",RECOVERABLE);

#ifdef MPI_EVOLVER
  if ( this_task == MASTER_TASK )
  { mpi_convert_to_quantities();
    outstring("Converting to all named quantities...");
  }
#else
  outstring("Converting to all named quantities...");
#endif

  /* set up default length or area quantities */
  if ( web.representation == STRING )
  {  q = new_quantity("default_length",Q_ENERGY);
     default_area_quant_num = q;
  }
  else q = new_quantity("default_length",Q_INFO);
  GEN_QUANT(q)->flags |= DEFAULT_QUANTITY; 
  
  if ( length_method_name[0] )
  { meth = find_method_instance(length_method_name);  /* user instance? */
    if ( meth < 0 )  /* try pre-defined method */
    { meth = new_method_instance(length_method_name,"default_length_inst");
      METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
    }
  }
  else if ( web.dimension >= 3 )
  { /* can't do edges in simplex model */
    meth = -1;
  }
  else if ( klein_metric_flag )
  { meth = new_method_instance("klein_length","default_length_inst");
    METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
  }
  else if ( web.conformal_flag )
  {  int oldflag = datafile_flag;
     meth = new_method_instance("edge_scalar_integral","default_length_inst");
     METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
     strcpy(formula,"density*sqrt(");
     gformula = print_express(&web.metric[0][0],'X');
     if ( strlen(gformula) > sizeof(formula)-20 )
       kb_error(1755,"Conformal metric formula too long.\n",RECOVERABLE);

     strcat(formula,gformula);
     strcat(formula,")");
     METH_INSTANCE(meth)->expr[0] = 
          (struct expnode *)mycalloc(1,sizeof(struct expnode));
     cmdptr = formula;
     datafile_flag = 1; 
     exparse(SDIM,METH_INSTANCE(meth)->expr[0],USERCOPY); line_no--;
     sprintf(METH_INSTANCE(meth)->expr[0]->name,"conformal metric");
     cmdptr = NULL; datafile_flag = oldflag;
  }
  else if ( web.metric_flag )
  {  int oldflag = datafile_flag;
     meth = new_method_instance("edge_general_integral","default_length_inst");
     METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
     strcpy(formula,"density*sqrt(");
     for ( i = 1 ; i <= SDIM ; i++ )
       for ( j = 1 ; j <= i  ; j++ )
       { if ( i > 1 ) strcat(formula,"+");
          if ( i==j ) sprintf(formula+strlen(formula),"X%d^2*(",i+SDIM);
          else sprintf(formula+strlen(formula),"2*X%d*X%d*(",i+SDIM,j+SDIM);
          gformula = print_express(&web.metric[i-1][j-1],'X');
          if ( strlen(gformula) > sizeof(formula)-20 )
            kb_error(1756,"Metric formula too long.\n",RECOVERABLE);
          strcat(formula,gformula);
          strcat(formula,")");
       }
     strcat(formula,")");
     METH_INSTANCE(meth)->expr[0] = 
         (struct expnode *)mycalloc(1,sizeof(struct expnode));
     cmdptr = formula;
     datafile_flag = 1;
     exparse(2*SDIM,METH_INSTANCE(meth)->expr[0],USERCOPY); line_no--;
     sprintf(METH_INSTANCE(meth)->expr[0]->name,"conformal metric");
     cmdptr = NULL; datafile_flag = oldflag;
  }
  else if ( web.representation == STRING )
  { meth = new_method_instance("density_edge_length","default_length_inst");
    METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
  }
  else
  { meth = new_method_instance("edge_length","default_length_inst");
    METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
  }
  length_method_number = meth;
  if ( meth >= 0 )
  {
    METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE;
    if ( web.representation == STRING )
      METH_INSTANCE(meth)->flags |= USE_DENSITY;
    attach_method_num(q,meth);
    apply_method_num(NULLID,meth);  /* global method */
  }
  
  if ( web.representation != STRING )
  /* soapfilm or simplex */
  {  q = new_quantity("default_area",Q_ENERGY);
     default_area_quant_num = q;
     GEN_QUANT(q)->flags |= DEFAULT_QUANTITY;
     if ( area_method_name[0] )
     { meth = find_method_instance(area_method_name);  /* user instance? */
       if ( meth < 0 )  /* try pre-defined method */
       { meth = new_method_instance(area_method_name,"default_area_inst");
         METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
       }
     }
     else if ( web.wulff_flag )
     { meth = new_method_instance("wulff_energy","default_area_inst");
       METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
     }
     else if ( klein_metric_flag )
     { meth = new_method_instance("klein_area","default_area_inst");
       METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
     }
     else if ( web.conformal_flag )
     {  int oldflag = datafile_flag;
        meth = new_method_instance("facet_scalar_integral","default_area_inst");
        METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
        strcpy(formula,"density*(");
        gformula = print_express(&web.metric[0][0],'X');
        if ( strlen(gformula) > sizeof(formula)-20 )
          kb_error(1757,"Conformal metric formula too long.\n",RECOVERABLE);

        strcat(formula,gformula);
        strcat(formula,")");
        METH_INSTANCE(meth)->expr[0] = 
            (struct expnode *)mycalloc(1,sizeof(struct expnode));
        cmdptr = formula;
        datafile_flag = 1;
        exparse(SDIM,METH_INSTANCE(meth)->expr[0],USERCOPY); line_no--;
        sprintf(METH_INSTANCE(meth)->expr[0]->name,"conformal metric");
        cmdptr = NULL; datafile_flag = oldflag;
     }
     else if ( web.metric_flag )
     {
       meth = new_method_instance("metric_facet_area","default_area_inst");
       METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
     }
     else
     { meth = new_method_instance("density_facet_area","default_area_inst");
       METH_INSTANCE(meth)->flags |= DEFAULT_INSTANCE;
     }
     METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE;
     METH_INSTANCE(meth)->flags |= USE_DENSITY;
     attach_method_num(q,meth);
     apply_method_num(NULLID,meth);  /* global method */
  }

  /* constraint energy integrals */
  for ( i = 0 ; i < web.maxcon ; i++ )
    convert_constraint_to_quantities(i);



  /* boundary energy integrals */
  for ( i = 0 ; i < web.bdrymax ; i++ )
  { struct boundary *bdry = web.boundaries + i;
    if ( !(bdry->attr & CON_ENERGY) ) continue;
    if ( bdry->attr & NAMED_THING )
    { sprintf(qname,"boundary_%s_energy",bdry->name);
      sprintf(inst_name,"boundary_%s_energy_inst",bdry->name);
    } else
    { sprintf(qname,"boundary_%d_energy",i);
      sprintf(inst_name,"boundary_%d_energy_inst",i);
    }
    gq = new_quantity(qname,Q_ENERGY);
    GEN_QUANT(gq)->flags |= DEFAULT_QUANTITY;
    if ( web.representation ==  STRING )
        meth = new_method_instance("vertex_scalar_integral",inst_name);
    else
        meth = new_method_instance("edge_vector_integral",inst_name);
    METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE|DEFAULT_INSTANCE;
    attach_method_num(gq,meth);
    { vertex_id v_id;
      if ( web.representation == STRING )
      { FOR_ALL_VERTICES(v_id)
          if ( bdry == get_boundary(v_id) ) 
             apply_quantity(v_id,gq);
      }
      else 
      { FOR_ALL_EDGES(e_id)
           if ( bdry == get_edge_boundary(e_id) ) 
              apply_quantity(e_id,gq);
      }
    }
    for ( j = 0 ; j < SDIM ; j++ ) 
       METH_INSTANCE(meth)->expr[j] = bdry->envect[j];
    bdry->attr |= USURPED_BY_QUANTITY;  /* prevent dual expr deallocation */
    bdry->energy_method = meth;
  }

  /* bodies */
  FOR_ALL_BODIES(b_id)
    convert_new_body_to_quantity(b_id);  /* set up individual stuff */
  convert_bodies_to_quantities();  /* add quantities to edges and facets */

  /* gravity */
  /* if ( web.gravflag ) do it always anyway! */
  {
     sprintf(qname,"gravity_quant");
     sprintf(inst_name,"gravity_inst");
     gq = new_quantity(qname,Q_ENERGY);
     GEN_QUANT(gq)->flags |= DEFAULT_QUANTITY;
     GEN_QUANT(gq)->modulus = web.grav_const;
     gravity_quantity_num = gq;
     if ( web.representation ==  STRING )
        meth = new_method_instance("string_gravity",inst_name);
     else
        meth = new_method_instance("gravity_method",inst_name);
     METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE|DEFAULT_INSTANCE;
     attach_method_num(gq,meth);
     apply_method_num(NULLID,meth);
  }

  if ( mean_curv_int_flag )
     mean_curv_int_quantity_num = 
         add_standard_quantity("mean_curvature_integral",1.0);

  if ( sqgauss_flag )
     add_standard_quantity("sq_gauss_curvature",1.0);

  /* square mean curvature */
  if ( square_curvature_flag )
  {
     sprintf(qname,"sq_mean_curvature_quant");
     sprintf(inst_name,"sq_mean_curvature_inst");
     gq = new_quantity(qname,Q_ENERGY);
     GEN_QUANT(gq)->flags |= DEFAULT_QUANTITY;
     sq_mean_curv_quantity_num = gq;
     if ( web.representation ==  STRING )
        meth = new_method_instance("sqcurve_string",inst_name);
     else
        meth = new_method_instance("sq_mean_curvature",inst_name);
     METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE|DEFAULT_INSTANCE;
     attach_method_num(gq,meth);
     apply_method_num(NULLID,meth);
     GEN_QUANT(gq)->modulus = globals(square_curvature_param)->value.real;
  }

  if ( web.convex_flag )
  {
     sprintf(qname,"gap_quant");
     sprintf(inst_name,"gap_energy");
     gq = new_quantity(qname,Q_ENERGY);
     GEN_QUANT(gq)->flags |= DEFAULT_QUANTITY;
     gap_quantity_num = gq;
     meth = new_method_instance("gap_energy",inst_name);
     METH_INSTANCE(meth)->flags |= IMPLICIT_INSTANCE|DEFAULT_INSTANCE;
     attach_method_num(gq,meth);
     apply_method_num(NULLID,meth);
  }

  quantities_only_flag = everything_quantities_flag = 1;

  outstring("Done.\n");
} // end convert_to_quantities()


/****************************************************************************
*
* function: convert_body_to_quantity()
*
* purpose: Convert a body volume to a named quantity. Done for
*             individual bodies so can be called when new body created,
*             i.e. rebody().
*/

void convert_body_to_quantity(body_id b_id)
{
  int gq;
  int meth1;
  char inst_name1[40];
  char qname[40];
  int i,j,k;
  edge_id e_id;
  facet_id f_id;
  facetedge_id fe;
  char formula[100];
  struct gen_quant *g = NULL;

  i = ordinal(b_id) + 1;
  sprintf(qname,"body_%d_vol",i);
  sprintf(inst_name1,"body_%d_vol_meth",i);
  if ( get_battr(b_id) & FIXEDVOL )
  { 
    gq = find_quantity(qname);
    if ( gq < 0 ) gq = new_quantity(qname,Q_FIXED);
    g = GEN_QUANT(gq);
    g->target = get_body_fixvol(b_id);
  }
  else if ( get_battr(b_id) & PRESSURE )
  { REAL p;
    gq = find_quantity(qname);
    if ( gq < 0 ) gq = new_quantity(qname,Q_ENERGY);
    g = GEN_QUANT(gq);
    p = get_body_pressure(b_id);
    g->modulus = -p;
    if ( p == 0.0 )
    { g->flags &= ~Q_ENERGY;
      g->flags |= Q_INFO; /* to force volume calc even for zero pressure */
      g->modulus = 1.0;
    }
  }
  else { gq = find_quantity(qname);
         if ( gq < 0 ) gq = new_quantity(qname,Q_INFO);
         g = GEN_QUANT(gq);
       }
  g->flags |= DEFAULT_QUANTITY;
  g->b_id = b_id;
  set_body_volquant(b_id,gq);
  g->volconst = get_body_volconst(b_id);
  g->pressure = get_body_pressure(b_id);
  g->value = get_body_volume(b_id);
  g->oldvalue = get_body_volume(b_id);

  if ( web.representation == STRING )
  {
     meth1 = find_method_instance(inst_name1);
     if ( meth1 < 0 )
     { /* have to create */
        if ( area_method_name[0] )
        { meth1 = find_method_instance(area_method_name);  /* user instance? */
          if ( meth1 < 0 )  /* try pre-defined method */
          { meth1 = new_method_instance(area_method_name,inst_name1);
            METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
          }
        }
        else if ( web.symmetric_content )
        { 
          meth1 = new_method_instance("edge_vector_integral",inst_name1);
          METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
          METH_INSTANCE(meth1)->expr[0] = 
              (struct expnode *)mycalloc(SDIM,sizeof(struct expnode));
          for ( j = 0 ; j < 2 ; j++ )
          { if ( j==0 )sprintf(formula,"-y/2");
            if ( j==1 )sprintf(formula,"x/2");
            cmdptr = formula;
            METH_INSTANCE(meth1)->expr[j] = METH_INSTANCE(meth1)->expr[0] + j;
            datafile_flag = 1;
            exparse(SDIM,METH_INSTANCE(meth1)->expr[j],USERCOPY); line_no--;
            sprintf(METH_INSTANCE(meth1)->expr[j]->name,
                "symmetric content component %d",j+1);
            datafile_flag = 0;
            cmdptr = NULL;
          }
        }
        else
        {
          meth1 = new_method_instance("edge_area",inst_name1);
          METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
        }
        METH_INSTANCE(meth1)->flags |= BODY_INSTANCE;
        METH_INSTANCE(meth1)->modulus = 1.0;
        attach_method_num(gq,meth1);
        set_body_volmeth(b_id,meth1);
     }
     FOR_ALL_EDGES(e_id)
     { fe = get_edge_fe(e_id); 
        if ( !valid_id(fe)  ) continue;
        if ( get_eattr(e_id) & NONCONTENT ) continue;
        f_id = get_fe_facet(fe);
        if ( valid_id(f_id)) 
        { if ( equal_id(b_id,get_facet_body(f_id)) )
             apply_method_num(e_id,meth1);
          else if ( equal_id(b_id,get_facet_body(inverse_id(f_id))) )
             apply_method_num(inverse_id(e_id),meth1);
        }
        fe = get_next_facet(fe);
        f_id = get_fe_facet(fe);
        if ( valid_id(f_id)) 
        { /* check both in case of torus wrapping */
          if ( equal_id(b_id,get_facet_body(f_id)) )
             apply_method_num(e_id,meth1);
          if ( equal_id(b_id,get_facet_body(inverse_id(f_id))) )
             apply_method_num(inverse_id(e_id),meth1);
        }
     }
  }
  else  /* SOAPFILM */
  {
     meth1 = find_method_instance(inst_name1);
     if ( meth1 < 0 )
     { /* have to create */
        if ( volume_method_name[0] )
        { meth1 = find_method_instance(volume_method_name); /* user instance? */
          if ( meth1 < 0 )  /* try pre-defined method */
          { meth1 = new_method_instance(volume_method_name,inst_name1);
            METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
          }
        }
        else if ( web.symmetric_content )
        {
          meth1 = new_method_instance("facet_vector_integral",inst_name1);
          METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
          METH_INSTANCE(meth1)->expr[0] = 
            (struct expnode *)mycalloc(SDIM,sizeof(struct expnode));
          for ( j = 0 ; j < SDIM ; j++ )
          { sprintf(formula,"x%d/%d",j+1,SDIM);
            cmdptr = formula;
            METH_INSTANCE(meth1)->expr[j] = METH_INSTANCE(meth1)->expr[0] + j;
            datafile_flag = 1;
            exparse(SDIM,METH_INSTANCE(meth1)->expr[j],USERCOPY); line_no--;
            sprintf(METH_INSTANCE(meth1)->expr[j]->name,
               "symmetric content component %d",j+1);
            datafile_flag = 0;
            cmdptr = NULL;
          }
        }
        else
        {
          meth1 = new_method_instance("facet_volume",inst_name1);
          METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
        }
        METH_INSTANCE(meth1)->flags |= BODY_INSTANCE;
        METH_INSTANCE(meth1)->modulus = 1.0;
        attach_method_num(gq,meth1);
        set_body_volmeth(b_id,meth1);
     }
     FOR_ALL_FACETS(f_id)
     { /* check both in case of torus wrapping */
        if ( get_fattr(f_id) & NONCONTENT ) continue;
        if ( equal_id(b_id,get_facet_body(f_id)) )
            apply_method_num(f_id,meth1);
        if ( equal_id(b_id,get_facet_body(inverse_id(f_id))) )
            apply_method_num(inverse_id(f_id),meth1);
     }
  }

  /* constraint content integrals */
  for ( j = 0 ; j < web.maxcon ; j++ )
  { struct constraint *con = get_constraint(j);
    vertex_id v_id;
    struct method_instance *mi;

    if ( !(con->attr & CON_CONTENT) ) continue;
    if ( con->attr & NAMED_THING )
      sprintf(inst_name1,"body_%d_%s_meth",i,con->name);
    else
      sprintf(inst_name1,"body_%d_con_%d_meth",i,j);
    meth1 = find_method_instance(inst_name1);
    if ( meth1 < 0 )
    { /* have to create */
      if ( web.representation ==  STRING )
      { meth1 = new_method_instance("vertex_scalar_integral",inst_name1);
      }
      else
      { meth1 = new_method_instance("edge_vector_integral",inst_name1);
      }
    }
    mi = METH_INSTANCE(meth1);
    mi->flags |= DEFAULT_INSTANCE | BODY_INSTANCE;
    mi->connum = j; 
    attach_method_num(gq,meth1);
    con = get_constraint(j); /* may have moved */
    if ( web.representation == STRING )
      mi->expr[0]  = con->convect[0];
    else
      for ( k = 0 ; k < SDIM ; k++ ) 
        mi->expr[k] = con->convect[k];
    
    if ( web.representation == STRING )
    { FOR_ALL_VERTICES(v_id)
      if ( v_on_constraint(v_id,j) ) 
      { edge_id first_e = e_id = get_vertex_edge(v_id);
        int max_rank,min_rank,jj;
        conmap_t *map = get_v_constraint_map(v_id);

        min_rank = MAXINT; max_rank = 0;
        for ( jj = 1 ; jj <= (int)map[0] ; jj++ )
        { struct constraint *c;
          if ( !(map[j] & CON_HIT_BIT) ) continue;
          c = get_constraint(map[jj]);
          if ( c->content_rank < min_rank ) min_rank = c->content_rank;
          if ( c->content_rank > max_rank ) max_rank = c->content_rank;
        }

        if ( valid_id(e_id) )
        do
        { 
          facetedge_id first_fe = fe = get_edge_fe(e_id);
          if ( valid_id(fe) && !(get_eattr(e_id) & NONCONTENT) )  
          do
          { f_id = get_fe_facet(fe);
            if ( equal_id(b_id,get_facet_body(f_id)) && 
              ( (!inverted(f_id) && con->content_rank >= max_rank) 
               || (inverted(f_id) && con->content_rank <= min_rank)) )
              apply_method_num(inverse_id(v_id),meth1);
            if ( equal_id(b_id,get_facet_body(inverse_id(f_id))) && 
              ( (!inverted(f_id) && con->content_rank >= max_rank) 
               || (inverted(f_id) && con->content_rank <= min_rank)) )
              apply_method(v_id,inst_name1);
            fe = get_next_facet(fe);
          } while ( !equal_id(fe,first_fe) );
          e_id = get_next_tail_edge(e_id);
        } while ( !equal_id(first_e,e_id));
      }
    }
    else  /* SOAPFILM */
    { FOR_ALL_EDGES(e_id)
      if ( e_on_constraint(e_id,j) ) 
      { facetedge_id first_fe = fe = get_edge_fe(e_id);
        
        if ( valid_id ( fe ) )
        { conmap_t *conmap = get_e_constraint_map(e_id);
          int min_rank = MAXINT; 
          int max_rank = 0;
          int jj;

          for ( jj = 1 ; jj <= (int)conmap[0] ; jj++ )
          { 
            struct constraint *c = get_constraint(conmap[jj]);
            if ( c->content_rank < min_rank ) min_rank = c->content_rank;
            if ( c->content_rank > max_rank ) max_rank = c->content_rank;
          }
          do
          { f_id = get_fe_facet(fe);
            if ( !(get_fattr(f_id) & NONCONTENT) )
            { if ( equal_id(b_id,get_facet_body(f_id)) 
                       && (con->content_rank >= max_rank) )
                apply_method_num(e_id,meth1);
              if ( equal_id(b_id,get_facet_body(inverse_id(f_id))) 
                       && (con->content_rank <= min_rank) )
                apply_method_num(inverse_id(e_id),meth1);
            }
            fe = get_next_facet(fe);
          } while ( !equal_id(fe,first_fe) );
        }
      }
    }
  }

  /* boundary content integrals */
  for ( j = 0 ; j < web.bdrymax ; j++ )
  { struct boundary *bdry = web.boundaries + j;
    vertex_id v_id;

    if ( !(bdry->attr & CON_CONTENT) ) continue;
    sprintf(inst_name1,"body_%d_bdry_%s_meth",i+1,bdry->name);
    meth1 = find_method_instance(inst_name1);
    if ( meth1 < 0 )
    { /* have to create */
      if ( web.representation ==  STRING )
      { meth1 = new_method_instance("vertex_scalar_integral",inst_name1);
      }
      else
      { meth1 = new_method_instance("edge_vector_integral",inst_name1);
      }
    }
    METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE | BODY_INSTANCE;
    METH_INSTANCE(meth1)->connum = j;
    attach_method_num(gq,meth1);
    if ( web.representation == STRING )
       METH_INSTANCE(meth1)->expr[0] = bdry->convect[0];
    else
       for ( k = 0 ; k < SDIM ; k++ ) 
           METH_INSTANCE(meth1)->expr[k] = bdry->convect[k];
     
     if ( web.representation == STRING )
     { FOR_ALL_VERTICES(v_id)
       if ( bdry == get_boundary(v_id) ) 
       { edge_id first_e = e_id = get_vertex_edge(v_id);
         if ( valid_id(e_id) )
         do
         { facetedge_id first_fe = fe = get_edge_fe(e_id);
           if ( valid_id(fe) && !(get_eattr(e_id) & NONCONTENT) ) 
           do
           { f_id = get_fe_facet(fe);
             if ( equal_id(b_id,get_facet_body(f_id)) )
               apply_method_num(v_id,meth1);
             if ( equal_id(b_id,get_facet_body(inverse_id(f_id))) )
               apply_method_num(inverse_id(v_id),meth1);
             fe = get_next_facet(fe);
           } while ( !equal_id(fe,first_fe) );
           e_id = get_next_tail_edge(e_id);
         } while ( !equal_id(first_e,e_id));
       }
     }
     else  /* SOAPFILM */
     { FOR_ALL_EDGES(e_id)
       if ( bdry == get_edge_boundary(e_id)  ) 
       { facetedge_id first_fe = fe = get_edge_fe(e_id);
         if ( valid_id(fe) )
           do
           { f_id = get_fe_facet(fe);
             if ( !(get_fattr(f_id) & NONCONTENT) )
             { if ( equal_id(b_id,get_facet_body(f_id)) )
                 apply_method_num(e_id,meth1);
               if ( equal_id(b_id,get_facet_body(inverse_id(f_id))) )
                 apply_method_num(inverse_id(e_id),meth1);
             }
             fe = get_next_facet(fe);
           } while ( !equal_id(fe,first_fe) );
       }
     }
  }
  create_pressure_quant(b_id);
} /* end convert_body_to_quantity() */

/****************************************************************************
*
* function: convert_new_body_to_quantity()
*
* purpose: Convert a new body volume to a named quantity.  Does not
*             do anything about putting quantities on edges or faces.
*             To can be called when new body created.
*             Call convert_bodies_to_quantity() to apply quantities.
*/

void convert_new_body_to_quantity(body_id b_id)
{
  int gq;
  int meth1;
  char inst_name1[40];
  char qname[40];
  int i,j;
  char formula[100];
  struct gen_quant *g=NULL; 

  i = ordinal(b_id) + 1;
  sprintf(qname,"body_%d_vol",i);
  sprintf(inst_name1,"body_%d_vol_meth",i);
  if ( get_battr(b_id) & FIXEDVOL )
  { 
    gq = find_quantity(qname);
    if ( gq < 0 ) gq = new_quantity(qname,Q_FIXED);
    g = GEN_QUANT(gq);
    g->target = get_body_fixvol(b_id);
  }
  else if ( get_battr(b_id) & PRESSURE )
  { REAL p;
    gq = find_quantity(qname);
    if ( gq < 0 ) gq = new_quantity(qname,Q_ENERGY);
    g = GEN_QUANT(gq);
    p = get_body_pressure(b_id);
    g->modulus = -p;
    if ( p == 0.0 )
    { g->flags &= ~Q_ENERGY;
      g->flags |= Q_INFO; 
        /* to force volume calc even for zero pressure */
      g->modulus = 1.0;
    }
  }
  else { gq = find_quantity(qname);
         if ( gq < 0 ) 
            gq = new_quantity(qname,Q_INFO);
         g = GEN_QUANT(gq);
       }
  g->flags |= DEFAULT_QUANTITY;
  g->b_id = b_id;
  set_body_volquant(b_id,gq);
  g->volconst = get_body_volconst(b_id);
  g->pressure = get_body_pressure(b_id);
  g->value = get_body_volume(b_id);
  g->oldvalue = get_body_volume(b_id);

  if ( web.representation == STRING )
  {
     meth1 = find_method_instance(inst_name1);
     if ( meth1 < 0 )
     { /* have to create */
        if ( area_method_name[0] )
        { meth1 = dup_method_instance(area_method_name,inst_name1);  /* user instance? */
          if ( meth1 < 0 )  /* try pre-defined method */
          { meth1 = new_method_instance(area_method_name,inst_name1);
            METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
          }
        }
        else if ( web.symmetric_content )
        { 
          meth1 = new_method_instance("edge_vector_integral",inst_name1);
          METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
          METH_INSTANCE(meth1)->expr[0] = 
              (struct expnode *)mycalloc(SDIM,sizeof(struct expnode));
          for ( j = 0 ; j < SDIM ; j++ )
          { int old_datafile_flag;
            char *old_cmdptr = cmdptr;
            int old_tok = tok;
            if ( j==0 )sprintf(formula,"-y/2");
            else if ( j==1 )sprintf(formula,"x/2");
            else sprintf(formula,"0");
            cmdptr = formula;
            METH_INSTANCE(meth1)->expr[j] = METH_INSTANCE(meth1)->expr[0] + j;
            old_datafile_flag = datafile_flag;
            datafile_flag = 1;
            exparse(SDIM,METH_INSTANCE(meth1)->expr[j],USERCOPY); line_no--;
            sprintf(METH_INSTANCE(meth1)->expr[j]->name,
                "symmetric content component %d",j+1);
            datafile_flag = old_datafile_flag;
            cmdptr = old_cmdptr;
            tok = old_tok;
  
          }
        }
        else
        {
          meth1 = new_method_instance("edge_area",inst_name1);
          METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
        }

     }
             
     METH_INSTANCE(meth1)->flags |= BODY_INSTANCE;
     METH_INSTANCE(meth1)->modulus = 1.0;
     attach_method_num(gq,meth1);
     set_body_volmeth(b_id,meth1);
  }
  else  /* SOAPFILM */
  {
     meth1 = find_method_instance(inst_name1);
     if ( meth1 < 0 )
     { /* have to create */
        if ( volume_method_name[0] )
        { meth1 = dup_method_instance(volume_method_name,inst_name1); /* user instance? */
          if ( meth1 < 0 )  /* try pre-defined method */
          { meth1 = new_method_instance(volume_method_name,inst_name1);
            METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
          }
        }
        else if ( web.symmetric_content )
        { struct method_instance *mi;
          meth1 = new_method_instance("facet_vector_integral",inst_name1);
          mi = METH_INSTANCE(meth1);
          mi->flags |= DEFAULT_INSTANCE;
          mi->expr[0] = 
            (struct expnode *)mycalloc(SDIM,sizeof(struct expnode));
          for ( j = 0 ; j < SDIM ; j++ )
          { int old_datafile_flag;
            char *old_cmdptr = cmdptr;
            int old_tok = tok;
            sprintf(formula,"x%d/%d",j+1,SDIM);
            cmdptr = formula;
            mi->expr[j] = mi->expr[0] + j;
            old_datafile_flag = datafile_flag;
            datafile_flag = 1;
            exparse(SDIM,mi->expr[j],USERCOPY); line_no--;
            sprintf(mi->expr[j]->name,"symmetric content component %d",j+1);
            datafile_flag = old_datafile_flag;
            cmdptr = old_cmdptr;
            tok = old_tok;
          }
        }
        else
        {
          meth1 = new_method_instance("facet_volume",inst_name1);
          METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE;
        }
        METH_INSTANCE(meth1)->flags |= BODY_INSTANCE;
        METH_INSTANCE(meth1)->modulus = 1.0;
        attach_method_num(gq,meth1);    
     }
     set_body_volmeth(b_id,meth1);
  }
  create_pressure_quant(b_id);
} /* end convert_new_body_to_quantity() */

/****************************************************************************
*
* function: create_body_constraint_content_method()
*
* purpose: Create a new method instance for constraint content.
*
* return: method number
*/
int create_body_constraint_content_method(
  body_id b_id,
  int connum)
{ struct constraint *con = get_constraint(connum);
  int k;
  int meth1;
  char inst_name1[100];
    
  if ( con->attr & NAMED_THING )
    sprintf(inst_name1,"body_%d_%s_meth",ordinal(b_id)+1,con->name);
  else
    sprintf(inst_name1,"body_%d_con_%d_meth",ordinal(b_id)+1,connum);
  meth1 = find_method_instance(inst_name1);
  if ( meth1 < 0 )
  { /* have to create */
    if ( web.representation ==  STRING )
    { meth1 = new_method_instance("vertex_scalar_integral",inst_name1);
    }
    else
    { meth1 = new_method_instance("edge_vector_integral",inst_name1);
    }
    METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE | BODY_INSTANCE;
    METH_INSTANCE(meth1)->connum = connum;
    con = get_constraint(connum); /* may have moved */
    if ( web.representation == STRING )
       METH_INSTANCE(meth1)->expr[0]  = con->convect[0];
    else
       for ( k = 0 ; k < SDIM ; k++ ) 
           METH_INSTANCE(meth1)->expr[k] = con->convect[k];
    attach_method_num(get_body_volquant(b_id),meth1);
  }
  return meth1;
} // end create_body_constraint_content_method()

/****************************************************************************
*
* function: create_body_boundary_content_method()
*
* purpose: Create a new method instance for boundary content.
*
* return: method number
*/
int create_body_boundary_content_method(
  body_id b_id,
  int bdrynum)
{ struct boundary *bdry = web.boundaries + bdrynum;
  int k;
  int meth1;
  char inst_name1[100];
 
      sprintf(inst_name1,"body_%d_bdry_%s_meth",ordinal(b_id)+1,bdry->name);
      meth1 = find_method_instance(inst_name1);
      if ( meth1 < 0 )
      { /* have to create */
        if ( web.representation ==  STRING )
        { meth1 = new_method_instance("vertex_scalar_integral",inst_name1);
        }
        else
        { meth1 = new_method_instance("edge_vector_integral",inst_name1);
        }
        METH_INSTANCE(meth1)->flags |= DEFAULT_INSTANCE | BODY_INSTANCE;
        if ( web.representation == STRING )
           METH_INSTANCE(meth1)->expr[0] = bdry->convect[0];
        else
          for ( k = 0 ; k < SDIM ; k++ ) 
           METH_INSTANCE(meth1)->expr[k] = bdry->convect[k];
        attach_method_num(get_body_volquant(b_id),meth1);
      }
    return meth1;
} // end create_body_boundary_content_method()

/****************************************************************************
*
* function: convert_bodies_to_quantities()
*
* purpose: Apply all body quantity method instances to elements.
*/

void convert_bodies_to_quantities()
{
  body_id b_id;
  int meth1;
  char inst_name1[100];
  int i,j,k;
  edge_id e_id;
  facet_id f_id;
  facetedge_id fe;

  /* The main surface integral */
  if ( web.representation == STRING )
  {
    FOR_ALL_EDGES(e_id)
    { if ( get_eattr(e_id) & NONCONTENT ) 
        continue;
      check_edge_vol_methods(e_id);
#ifdef ZZZZ
      fe = get_edge_fe(e_id); 
      if ( !valid_id(fe)  ) continue;
      f_id = get_fe_facet(fe);
      for ( i = 0 ; i < 2 ; i++ ) /* at most two facets per edge */
      { if ( valid_id(f_id)) 
        { b_id = get_facet_body(f_id);
          if ( valid_id(b_id) )
             apply_method_num(e_id,get_body_volmeth(b_id));
          b_id = get_facet_body(inverse_id(f_id));
          if ( valid_id(b_id) )
             apply_method_num(inverse_id(e_id),get_body_volmeth(b_id));
        }
        fe = get_next_facet(fe);
        f_id = get_fe_facet(fe);
      }
#endif
    }
  }
  else  /* SOAPFILM */
  {
     FOR_ALL_FACETS(f_id)
     { /* check both in case of torus wrapping */
       if ( get_fattr(f_id) & NONCONTENT ) continue;
       b_id = get_facet_body(f_id);
       if ( valid_id(b_id) )
            apply_method_num(f_id,get_body_volmeth(b_id));
       b_id = get_facet_body(inverse_id(f_id));
       if ( valid_id(b_id) )
            apply_method_num(inverse_id(f_id),get_body_volmeth(b_id));
     }
  }

  /* constraint content integrals */

  /* create instance for each possible body/constraint content pair */
  for ( j = 0 ; j < web.maxcon ; j++ )
  { struct constraint *con = get_constraint(j);

    if ( !(con->attr & CON_CONTENT) ) continue;
    FOR_ALL_BODIES(b_id)
    { struct method_instance *mi;
      if ( con->attr & NAMED_THING )
        sprintf(inst_name1,"body_%d_%s_meth",ordinal(b_id)+1,con->name);
      else
        sprintf(inst_name1,"body_%d_con_%d_meth",ordinal(b_id)+1,j);
      meth1 = find_method_instance(inst_name1);
      if ( meth1 < 0 )
      { /* have to create */     
        if ( web.representation ==  STRING )
        { meth1 = new_method_instance("vertex_scalar_integral",inst_name1);
        }
        else
        { meth1 = new_method_instance("edge_vector_integral",inst_name1);
        }
      }
      mi = METH_INSTANCE(meth1);
      mi->flags |= DEFAULT_INSTANCE | BODY_INSTANCE;
      METH_INSTANCE(meth1)->connum = j;
      con = get_constraint(j); /* may have moved */
      if ( web.representation == STRING )
         mi->expr[0]  = con->convect[0];
      else
         for ( k = 0 ; k < SDIM ; k++ ) 
            mi->expr[k] = con->convect[k];
      
    }
  }
  if ( web.representation == STRING )
  { vertex_id v_id;
    FOR_ALL_VERTICES(v_id)
    { conmap_t *con = get_v_constraint_map(v_id);
      int concount = (int)con[0];
      edge_id first_e;
      int max_rank,min_rank;
        
      min_rank = MAXINT; max_rank = 0;
      for ( j = 1 ; j <= (int)con[0] ; j++ )
      { struct constraint *c;
        if ( !(con[j] & CON_HIT_BIT) ) continue;
        c = get_constraint(con[j]);
        if ( c->content_rank < min_rank ) min_rank = c->content_rank;
        if ( c->content_rank > max_rank ) max_rank = c->content_rank;
      }

      for ( i = 1 ; i <= concount ; i++ )
      { conmap_t *con = get_v_constraint_map(v_id); /* might have moved */
        int connum = con[i] & CONMASK;
        struct constraint *c = get_constraint(connum);
        if ( !(c->attr & CON_CONTENT)) continue;
        first_e = e_id = get_vertex_edge(v_id);
        if ( valid_id(e_id) )
        do
        { char name[100];
          facetedge_id first_fe = fe = get_edge_fe(e_id);
          if ( valid_id(fe) && !(get_eattr(e_id) & NONCONTENT) )  
          do
          { f_id = get_fe_facet(fe);
            fe = get_next_facet(fe);
            if ( !valid_id(f_id) ) continue;
            b_id = get_facet_body(f_id);
            if ( valid_id(b_id)   && 
           ( (!inverted(f_id) && c->content_rank >= max_rank) 
               || (inverted(f_id) && c->content_rank <= min_rank)))
            { if ( c->attr & NAMED_THING )
                sprintf(name,"body_%d_%s_meth",ordinal(b_id)+1,c->name);
              else
                sprintf(name,"body_%d_con_%d_meth",ordinal(b_id)+1,connum);
              attach_method(get_body_volquant(b_id),name);
              apply_method(inverse_id(v_id),name);
            }
            b_id = get_facet_body(inverse_id(f_id));
            if ( valid_id(b_id) && 
              ( (!inverted(f_id) && c->content_rank >= max_rank) 
               || (inverted(f_id) && c->content_rank <= min_rank)))
            { if ( c->attr & NAMED_THING )
                sprintf(name,"body_%d_%s_meth",ordinal(b_id)+1,c->name);
              else
                sprintf(name,"body_%d_con_%d_meth",ordinal(b_id)+1,connum);
              attach_method(get_body_volquant(b_id),name);
              apply_method(v_id,name);
            }
          } while ( !equal_id(fe,first_fe) );
          e_id = get_next_tail_edge(e_id);
        } while ( !equal_id(first_e,e_id));
      }
    }
  }
  else  /* SOAPFILM */
  { FOR_ALL_EDGES(e_id)
    { conmap_t *con = get_e_constraint_map(e_id);
      int concount = (int)con[0];
      facetedge_id first_fe = fe = get_edge_fe(e_id);
      if ( valid_id ( fe ) )
      { int min_rank, max_rank;

         min_rank = MAXINT; max_rank = 0;
         for ( j = 1 ; j <= concount ; j++ )
         { 
           struct constraint *c = get_constraint(con[j]);
           if ( c->content_rank < min_rank ) min_rank = c->content_rank;
           if ( c->content_rank > max_rank ) max_rank = c->content_rank;
         }
      for ( i = 1 ; i <= concount ; i++ )
      { conmap_t *con = get_e_constraint_map(e_id);  /* might have moved */
        int connum = con[i]&CONMASK;
        struct constraint *c = get_constraint(connum);
        if ( !(c->attr & CON_CONTENT) ) continue;
        fe = first_fe;
        do
        { char name[100];
          f_id = get_fe_facet(fe);
          fe = get_next_facet(fe);
          if ( !valid_id(f_id) ) continue;
          if ( get_fattr(f_id) & NONCONTENT ) continue;
          b_id = get_facet_body(f_id);
          if ( valid_id(b_id) && (c->content_rank >= max_rank))
          { if ( c->attr & NAMED_THING )
              sprintf(name,"body_%d_%s_meth",ordinal(b_id)+1,c->name);
            else
              sprintf(name,"body_%d_con_%d_meth",ordinal(b_id)+1,connum);
            attach_method(get_body_volquant(b_id),name);
            apply_method(e_id,name);
          }
          b_id = get_facet_body(inverse_id(f_id));
          if ( valid_id(b_id) && (c->content_rank <= min_rank) )
          { if ( c->attr & NAMED_THING )
              sprintf(name,"body_%d_%s_meth",ordinal(b_id)+1,c->name);
            else
              sprintf(name,"body_%d_con_%d_meth",ordinal(b_id)+1,connum&CONMASK);
            attach_method(get_body_volquant(b_id),name);
            apply_method(inverse_id(e_id),name);
          }
        } while ( !equal_id(fe,first_fe) );
      }
      }
    }
  }

  /* boundary content integrals */
  for ( j = 0 ; j < web.bdrymax ; j++ )
  { struct boundary *bdry = web.boundaries + j;

    if ( !(bdry->attr & CON_CONTENT) ) continue;
    FOR_ALL_BODIES(b_id)
    { struct method_instance *mi;
      sprintf(inst_name1,"body_%d_bdry_%s_meth",ordinal(b_id)+1,bdry->name);
      meth1 = find_method_instance(inst_name1);
      if ( meth1 < 0 )
      { /* have to create */
        if ( web.representation ==  STRING )
        { meth1 = new_method_instance("vertex_scalar_integral",inst_name1);
        }
        else
        { meth1 = new_method_instance("edge_vector_integral",inst_name1);
        }
      }
      mi = METH_INSTANCE(meth1);
      mi->flags |= DEFAULT_INSTANCE | BODY_INSTANCE;
      if ( web.representation == STRING )
         mi->expr[0] = bdry->convect[0];
      else
        for ( k = 0 ; k < SDIM ; k++ ) 
           mi->expr[k] = bdry->convect[k];
      attach_method_num(get_body_volquant(b_id),meth1);
    }
  }
  if ( web.representation == STRING )
  { vertex_id v_id;
    FOR_ALL_VERTICES(v_id)
    { struct boundary *bdry = get_boundary(v_id);  
      edge_id first_e;
      if ( !bdry || !(bdry->attr & CON_CONTENT) ) continue;
      first_e = e_id = get_vertex_edge(v_id);
      if ( valid_id(e_id) )
      do
      { facetedge_id first_fe = fe = get_edge_fe(e_id);
        if ( valid_id(fe) && !(get_eattr(e_id) & NONCONTENT) ) 
        do
        { f_id = get_fe_facet(fe);
          b_id = get_facet_body(f_id);
          if ( valid_id(b_id) )
          {
            sprintf(inst_name1,"body_%d_bdry_%s_meth",ordinal(b_id)+1,bdry->name);
            meth1 = find_method_instance(inst_name1);
            apply_method_num(v_id,meth1);
          }
          b_id = get_facet_body(inverse_id(f_id));
          if ( valid_id(b_id) )
          {
            sprintf(inst_name1,"body_%d_bdry_%s_meth",ordinal(b_id)+1,bdry->name);
            meth1 = find_method_instance(inst_name1);
            apply_method_num(inverse_id(v_id),meth1);
          }
          fe = get_next_facet(fe);
        } while ( !equal_id(fe,first_fe) );
        e_id = get_next_tail_edge(e_id);
      } while ( !equal_id(first_e,e_id));
    }
  }
  else  /* SOAPFILM */
  { FOR_ALL_EDGES(e_id)
    { struct boundary *bdry = get_edge_boundary(e_id);  
      facetedge_id first_fe;
      if ( !bdry || !(bdry->attr & CON_CONTENT) ) continue;
      first_fe = fe = get_edge_fe(e_id);
      if ( valid_id(fe) )
      do
      { f_id = get_fe_facet(fe);
        fe = get_next_facet(fe);
        if ( !valid_id(f_id) ) continue;
        if ( get_fattr(f_id) & NONCONTENT ) continue;
        b_id = get_facet_body(f_id);
        
        if ( valid_id(b_id) )
        { sprintf(inst_name1,"body_%d_bdry_%s_meth",ordinal(b_id)+1,bdry->name);
          meth1 = find_method_instance(inst_name1);
          apply_method_num(inverse_id(e_id),meth1);
        }
        b_id = get_facet_body(inverse_id(f_id));
        if ( valid_id(b_id) )
        { sprintf(inst_name1,"body_%d_bdry_%s_meth",ordinal(b_id)+1,bdry->name);
          meth1 = find_method_instance(inst_name1);
          apply_method_num(e_id,meth1);
        }
      } while ( !equal_id(fe,first_fe) );
    }
  }

  /* done in convert_new_body...
  if ( web.pressure_flag )
    FOR_ALL_BODIES(b_id)
      create_pressure_quant(b_id);
   */
} // end convert_bodies_to_quantities()

/***************************************************************************
*
*  function: create_pressure_quant()
*
*  purpose: create a named quantity for the ambient pressure energy.
*/
void create_pressure_quant(body_id b_id)
{
  if ( web.pressure_flag )
  { /* create formula and do compound quantity */
    int gq = get_body_volquant(b_id);
    int mc = GEN_QUANT(gq)->method_count;
    char *volsum = temp_calloc(40,mc);
    char *formula = temp_calloc(2,40*mc + 100);
    char qname[1000];
    int q,i,j;

    sprintf(qname,"body_%d_ambient_energy",ordinal(b_id)+1);
    q = find_quantity(qname);
    if ( q < 0 ) q = new_quantity(qname,Q_INFO);
    set_body_ambquant(b_id,q);
    GEN_QUANT(q)->flags |= DEFAULT_QUANTITY;
    sprintf(volsum,"body[%d].volconst",ordinal(b_id)+1);
    for ( j = 0 ; j < mc ; j++ )
    { int m = GEN_QUANT(gq)->meth_inst[j];
      struct method_instance *mi = METH_INSTANCE(m);
      strcat(volsum,((mi->modulus > 0.0) ? "+" : "+"));
      strcat(volsum,mi->name);
      strcat(volsum,".value");
      mi->flags |= Q_COMPOUND;
    }
    GEN_QUANT(q)->method_count = 0;
    i = ordinal(b_id)+1;
    sprintf(formula,"-ambient_pressure_value*(body[%d].target*(log(%s)-log(body[%d].target))-(%s-body[%d].target))",
       i,volsum,i,volsum,i);
    cmdptr = formula;
    exparse(0,&(GEN_QUANT(q)->expr),USERCOPY); line_no--;
    sprintf(msg,"%s formula",qname);
    strncpy(GEN_QUANT(q)->expr.name,msg,EXPNAMESIZE-1);
    cmdptr = NULL;
    GEN_QUANT(q)->flags |= Q_COMPOUND;
    GEN_QUANT(q)->next_compound = compound_quant_list_head;
    compound_quant_list_head = q;
    if ( get_battr(b_id) & FIXEDVOL ) 
       GEN_QUANT(q)->flags |= Q_ENERGY;
    else   GEN_QUANT(q)->flags |= Q_INFO;
    GEN_QUANT(gq)->flags &= ~Q_FIXED;
    GEN_QUANT(gq)->flags |= Q_INFO;
    temp_free(volsum);
    temp_free(formula);
  }  
} // end create_pressure_quant()

/*****************************************************************************
*
* Function: read_array_initializer()
*
* Purpose: Read bracketed array initial data.  Next unread token should
*          be the first '{'.
*/
void read_array_initializer(struct array *a)
{ char *spot;
    char *spots[MAXARRAYDIMS];
    int depth;
    int blocksize;
    int items[MAXARRAYDIMS];

    tok = yylex(); /* should be '{' */
    if ( tok != '{' )
      kb_error(3041,"Missing '{'; expected array initializer.\n",RECOVERABLE);       
 
    depth = 0;
    spot = (char *)a + a->datastart;
    spots[depth] = spot;
    blocksize = a->datacount; 
    for (;;)
    { 
      if ( depth == a->dim )
      { int k;
        for ( k = 0 ; ;  )
        { if ( read_single_value(a->datatype,spot) )
          { if ( k < a->sizes[depth-1] )
            { spot += a->itemsize;
              k++;
            }
            else kb_error(2132,"Too many initializers.\n",DATAFILE_ERROR);
          }          
          else
          { tok = yylex();  /* get back token exparse pushed back */
            if ( tok != ',' ) 
              break;
          }
        }
      }
      if ( tok == '{' )  
      { 
        if ( blocksize ) blocksize /= a->sizes[depth]; 
        spots[depth] = spot;
        items[depth]++;
        if ( depth > 0 && items[depth] > a->sizes[depth-1] )
        { kb_error(2539,"Too many initializers.\n",DATAFILE_ERROR);
          return;
        }
        depth++; 
        if ( depth > a->dim )
        { kb_error(7398,"Array initializer has too many dimensions.\n",DATAFILE_ERROR);
          return;
        }
        items[depth] = 0;
        if ( depth != a->dim ) tok = yylex();
      }
      else if ( tok == '}' )
      { tok = yylex();
        depth--;
        if ( depth == 0 ) return;

        blocksize *= a->sizes[depth];
        spots[depth] += blocksize*a->itemsize;
        spot = spots[depth];
      }
      else if ( tok == ',' )
      { tok = yylex();
        spots[depth] += blocksize*a->itemsize;
      }
      else 
      { sprintf(errmsg,"Illegal token \"%s\" in array initialization.\n",yytext);
        kb_error(2540,errmsg,DATAFILE_ERROR);
      }
    }
} /* end read_array_initializer() */

/****************************************************************************
*
* function: define_array()
*
* purpose: read an array parameter in the top of the datafile, 
*          with possible initialization.
*
* incoming assumption: array name in yytext.
*/

void define_array(void)
{ int n;
  struct global *g;
  struct array *a;
  int dim;
  int size;
  int sizes[MAXARRAYDIMS];
  int itemsize=0;
  int datatype=0;
  size_t rem;

  if ( strlen(yytext) < 2 )
    kb_error(1867,"Identifiers must be at least two characters long.\n",
            DATAFILE_ERROR);
  n = lookup_global(yytext);
  if ( n >= 0 )
  { 
    g = globals(n);
    if ( !(g->flags & ARRAY_PARAM) )
      kb_error(3213,"Array name already in use for something else.\n",
        DATAFILE_ERROR);
  }
  else
  { n = add_global(yytext);
    if ( n < 0 )
      kb_error(1673,"Duplicate parameter name.\n",DATAFILE_ERROR);
    g = globals(n);
    g->flags |= ARRAY_PARAM;
  }

  tok = yylex(); /* dispose of IDENT_ */
  if ( tok == DATATYPE_TOK )
  { datatype = yylval.datatype;
    itemsize = datatype_size[datatype];
  }
  else if ( tok == STRING_TOK ) // since "string" also used for model type
  { datatype = STRING_TYPE;
    itemsize = datatype_size[datatype];
  }
  else
  { kb_error(2130,"Need an array data type.  Assuming REAL.\n",WARNING);
    datatype = REAL_TYPE;
    itemsize = datatype_size[datatype];
  }
  g->type = datatype;

  tok = yylex(); /* dispose of type */

  g->attr.arrayptr = (struct array*)mycalloc(1,sizeof(struct array));
  g->attr.arrayptr->dim = 0;
  while ( tok == '[' )
  {  REAL val;
     if ( read_const(&val) < 0 )
       kb_error(1057,"Need dimension number.\n",DATAFILE_ERROR);
     sizes[g->attr.arrayptr->dim] = (int)val;
     if ( sizes[g->attr.arrayptr->dim] < 0 )
       kb_error(1059,"Array dimension cannot be negative.\n",DATAFILE_ERROR);
     tok = yylex(); /* get ] */
     if ( tok != ']' ) 
        kb_error(2131,"Expecting ].\n",DATAFILE_ERROR);
     tok = yylex(); /* eat ] */
     g->attr.arrayptr->dim++;
  }
  if ( g->attr.arrayptr->dim == 0 ) sizes[g->attr.arrayptr->dim++] = 1;

  for ( size = 1, dim=0 ; dim < g->attr.arrayptr->dim ; dim++ )
  { size *= sizes[dim];  }
  a = g->attr.arrayptr = (struct array*)kb_realloc((char*)(g->attr.arrayptr),
       sizeof(struct array)+dim*sizeof(int)
                  + (size+1)*itemsize);  /* extra for alignment */
  a->dim = g->attr.arrayptr->dim;
  a->itemsize = itemsize;
  a->datatype = datatype;
  a->datacount = size;
  for ( dim=0 ; dim < g->attr.arrayptr->dim ; dim++ )
     a->sizes[dim] = sizes[dim];
  a->datastart = sizeof(struct array) + a->dim*sizeof(int);

  /* align datastart to size of item */
  rem = a->datastart % a->itemsize;
  if ( rem )
     a->datastart += a->itemsize - rem;

  /* Initialization */
  if ( (tok == '=') || (tok == ASSIGN_TOK) ) /* have initial values */
    read_array_initializer(a);

} // end define_array()


/****************************************************************************
*
* Function: check_forwards()
*
* Purpose: Make sure all forward declarations have been implemented.
*/

void check_forwards()
{ int i;

  /* constraints */
  for ( i = 0 ; i < web.maxcon ; i++ )
  { struct constraint *con = get_constraint(i);
    if ( con->attr & CON_FORWARD_DEF )
    { sprintf(errmsg,"Forward declared constraint %s not instantiated.\n",
             con->name);
      kb_error(5911,errmsg,RECOVERABLE);
    }
  }

  /* boundaries */
  for ( i = 0 ; i < web.bdrymax ; i++ )
  { struct boundary *bdry = web.boundaries + i;
    if ( bdry->attr & BDRY_FORWARD_DEF )
    { sprintf(errmsg,"Forward declared boundary %s not instantiated.\n",
             bdry->name);
      kb_error(3227,errmsg,RECOVERABLE);
    }
  }

  /* quantities */
  for ( i = 0 ; i < gen_quant_count ; i++ )
  { if ( GEN_QUANT(i)->flags & Q_FORWARD_DEF )
    { sprintf(errmsg,"Forward declared quantity %s not instantiated.\n",
             GEN_QUANT(i)->name);
      kb_error(3225,errmsg,RECOVERABLE);
    }
  }

  /* method instances */
  for ( i = 1 ; i < meth_inst_count ; i++ )
  { if ( METH_INSTANCE(i)->flags & Q_FORWARD_DEF )
    { sprintf(errmsg,"Forward declared method_instance %s not instantiated.\n",
             METH_INSTANCE(i)->name);
      kb_error(3226,errmsg,RECOVERABLE);
    }
  }

} // end check_forwards()
