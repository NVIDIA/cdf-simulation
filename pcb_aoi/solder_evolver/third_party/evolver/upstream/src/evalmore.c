/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/


/*****************************************************************
*
*  File: evalmore.c
*
*  Purpose: Overflow from eval() in evaltree.c.
*/

#include "include.h" 
#include "lex.h"
#include "ytab.h"
#ifdef MSC
#include "psapi.h"
#endif

#ifdef MAC_APP
#define S_IFIFO 0x100
#elif !defined(MAC_CW)
#include <sys/stat.h>
/* #include <sys/wait.h> */
#ifndef S_IFIFO
#ifdef __S_IFIFO
#define S_IFIFO __S_IFIFO
#elif defined(_S_IFIFO)
#define S_IFIFO _S_IFIFO
#else
#define S_IFIFO 0010000
#endif
#endif
#endif
/*
int wait (int*);
int fork (void);
int execlp();
int stat();
*/


/************************************************************************
*
* Function: other_stuff()
*
* Purpose: execute command nodes, overflow from eval().
*
* Return: stacktop pointer
*/

void other_stuff(
  struct treenode * node,
  int *recalc_flag,
  int *update_display_flag,
  element_id q_id,
  REAL *localstack, /* stack base */
  struct locallist_t *localbase
)
{ int i,j,n,k;
  char *s,*h;
  REAL scale;
  int old; /* old state of toggle */
  int oldquiet; /* old state of quiet_flag */
  element_id id,fe,f_id,b_id;


  struct thread_data *td = GET_THREAD_DATA;
  #define newstack  (td->eval_stack)
  #define stackmax  (td->eval_stack_size)
  #define stacktop  (td->stack_top)
  #define this_frame ((struct eval_frame*)(newstack + td->frame_spot))

  switch ( node->type ) 
  {
    case CONSTRAINT_FIXED_NODE:
      { int connum = (int)(*stacktop--);
        struct constraint *con = get_constraint(connum);
        (*++stacktop) = !(con->attr & (NONPOSITIVE|NONNEGATIVE));
        break;
      }
    case CONSTRAINT_NONPOSITIVE_NODE:
      { int connum = (int)(*stacktop--);
        struct constraint *con = get_constraint(connum);
        (*++stacktop) = !!(con->attr & NONPOSITIVE);
        break;
      }
    case CONSTRAINT_NONNEGATIVE_NODE:
      { int connum = (int)(*stacktop--);
        struct constraint *con = get_constraint(connum);
        (*++stacktop) = !!(con->attr & NONNEGATIVE);
        break;
      }
    case CONSTRAINT_NORMAL_NODE:
      { int connum = (int)(*stacktop--);
        struct constraint *con = get_constraint(connum);
        (*++stacktop) = !!(con->attr & NONPOSITIVE);
        break;
      }
    case SIMPLEX_TO_FE_NODE:
       simplex_to_fe(); break; 
       
    case REORDER_STORAGE_NODE:
       reorder_storage(); break;

    case RENUMBER_ALL_NODE:
       renumber_all(); break;

    case DUMP_MEMLIST_NODE:
       mem_list_dump(); break;

    case VIEW_MATRIX_NODE:
       i = (int)(*(stacktop--));
       k = (int)(*(stacktop--));
       if ( (k < 1) || (k > SDIM+1) || (i < 1) || (i > SDIM+1) )
       { sprintf(errmsg,
           "Illegal index: view_matrix[%d][%d] (must be 1 to %d)\n",k,i,SDIM);
         sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
         kb_error(2008,errmsg,RECOVERABLE);
       }
       *(++stacktop) = view[k-1][i-1];
       break;

    case RESET_COUNTS_NODE:
       reset_counts();
       break;

    case FLUSH_COUNTS_NODE:
       flush_counts();
       break;

    case PRINT_PROFILING_NODE:
       print_profiling();
       break;

    case RESET_PROFILING_NODE:
       reset_profiling();
       break;

    case PAUSE_NODE:
       outstring("Paused; hit RETURN to continue, b to break: ");
       my_fgets(msg,msgmax,stdin);
       if ( msg[0] == 'b' ) 
          breakflag = BREAKFULL;
       break;

    case HELP_KEYWORD_NODE:
       keyword_help(node->op1.string);
       break;
  
    case CREATE_VERTEX_NODE:
     { vertex_id v_id;
       REAL x[MAXCOORD];
       for ( i = SDIM-1 ; i >= 0 ; i-- ) x[i] = *(stacktop--);
       v_id = new_vertex(x,NULLID);
       new_vertex_id = ordinal(v_id) + 1;
       set_original(v_id,new_vertex_id);
       *(++stacktop) = (REAL)new_vertex_id;
       *recalc_flag = 1;
       break;
     }

    case FACET_CROSSCUT_NODE:
      {
        int v2 = (int)(*stacktop--);
        int v1 = (int)(*stacktop--);
        int f1 = (int)(*stacktop--);
        edge_id new_edge_id = facet_crosscut(get_ordinal_id(FACET,f1-1),
                    get_ordinal_id(VERTEX,v1-1), get_ordinal_id(VERTEX,v2-1));
        *(++stacktop) = (REAL)(ordinal(new_edge_id)+1);
        *recalc_flag = 1;
        break;
      }

    case CREATE_EDGE_NODE:
      { vertex_id v_id2; 
        vertex_id v_id1; 
        int v2 = (int)(*stacktop--);
        int v1 = (int)(*stacktop--);
        edge_id e_id;
        v_id2 = get_ordinal_id(VERTEX,v2-1); 
        v_id1 = get_ordinal_id(VERTEX,v1-1); 
        if ( !valid_id(v_id1) )
        { sprintf(errmsg,"Invalid tail vertex %d in new_edge.\n",v1);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2009,errmsg,RECOVERABLE);
        }
        if ( !valid_id(v_id2) )
        { sprintf(errmsg,"Invalid head vertex %d in new_edge.\n",v2);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2010,errmsg, RECOVERABLE);
        }
        e_id = new_edge(v_id1,v_id2,NULLID);
        new_edge_id = ordinal(e_id)+1;
        set_original(e_id,new_edge_id);
        *(++stacktop) = (REAL)new_edge_id;
        *recalc_flag = 1;
      }
      break;

    case CREATE_FACET_NODE:
    { facetedge_id old_fe = NULLID;
       edge_id e_id;
       vertex_id tv,hv;
#define ORD_ID(id) ((id)<0 ? inverse_id(get_ordinal_id(EDGE,-1-(id)))  : \
                                   get_ordinal_id(EDGE,(id)-1))
       f_id = NULLID;
       fe = NULLID; 
       if ( (web.representation == STRING) || (web.representation == SOAPFILM) )
       { n = node[node->left].op1.argcount; /* size of edge list */
         for ( i = 0 ; i < n ; i++ ) 
           if ( !valid_id(ORD_ID((int)(stacktop[-i]))) )
           { sprintf(msg,"Invalid edge %g in new_facet.\n",(double)stacktop[-i]);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2011,msg,RECOVERABLE);
           }
         /* check continuity */
         i = 1-n;
         e_id = ORD_ID((int)(stacktop[i]));
         hv = get_edge_headv(e_id);
         for ( i++ ; i <= 0 ; i++ )
         { e_id = ORD_ID((int)(stacktop[i]));
           tv = get_edge_tailv(e_id);
           if ( !equal_id(hv,tv) )
           { sprintf(errmsg,
               "Inconsistency: edge %s tail vertex %s disagrees with previous head %s.\n",
                           ELNAME(e_id),ELNAME1(tv),ELNAME2(hv));
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2012,errmsg,RECOVERABLE);
           }
           hv = get_edge_headv(e_id);
         }
         if ( web.representation == SOAPFILM)
         {
           e_id = ORD_ID((int)(stacktop[0]));
           hv = get_edge_headv(e_id);
           e_id = ORD_ID((int)(stacktop[1-n]));
           tv = get_edge_tailv(e_id);
           if ( !equal_id(hv,tv) ) 
           { sprintf(errmsg,
             "Inconsistency in first edge tail vertex and last edge head.\n");
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2013,errmsg,RECOVERABLE);
           }
         }
         f_id = new_facet();
         set_original(f_id,f_id);
         for ( i = 0 ; i < n ; i++ ) 
         { facetedge_id edge_fe;
           e_id = ORD_ID((int)(stacktop[i-n+1]));
           fe = new_facetedge(f_id,e_id);
           if ( !valid_id(get_facet_fe(f_id)) )
              set_facet_fe(f_id,fe);
           else
           { set_next_edge(old_fe,fe);
             set_prev_edge(fe,old_fe);
           }
           old_fe = fe;
           /* add to edge facet list, and get in geometric order */
           edge_fe = get_edge_fe(e_id);
           if ( valid_id(edge_fe) )
           { /* insert in chain */
             set_next_facet(fe,get_next_facet(edge_fe));
             set_prev_facet(fe,edge_fe);
             set_prev_facet(get_next_facet(fe),fe);
             set_next_facet(edge_fe,fe);
           }
           else
           { set_next_facet(fe,fe);
             set_prev_facet(fe,fe);
             set_edge_fe(e_id,fe);      /* link edge to rest of world */
           }
         }
         if ( equal_id(get_fe_tailv(get_facet_fe(f_id)),get_fe_headv(fe)) )
         {
           set_next_edge(fe,get_facet_fe(f_id));  /* close up ring */
           set_prev_edge(get_facet_fe(f_id),fe);
         }
         if ( web.representation == SOAPFILM ) 
          { fe = get_facet_fe(f_id);
            for ( i = 0 ; i < n ; i++,fe = get_next_edge(fe) )
                fe_reorder(get_fe_edge(fe));
            if ( n > 3 ) face_triangulate(f_id,n);
            recalc_facet_area(f_id);
          }
         if ( web.representation == SOAPFILM ) 
           need_fe_reorder_flag = 1;

          stacktop -= n; /* pop id's */
       }
       else if ( web.representation == SIMPLEX )
       {  vertex_id *v;
          n = node[node->left].op1.argcount; /* size of edge list */
          for ( i = 0 ; i < n ; i++ )
            if ( !valid_id(get_ordinal_id(VERTEX,(int)stacktop[-i]-1) ) )
               { sprintf(msg,"Invalid vertex %d in new_facet.\n",(int)stacktop[-i]);
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                      file_names[node->file_no],node->line_no);
                 kb_error(2014,msg,RECOVERABLE);
               }
          f_id = new_facet();
          v = get_facet_vertices(f_id);
          for ( i = 0 ; i < n ; i++ ) v[i] = get_ordinal_id(VERTEX,(int)stacktop[-n+i+1]-1);
          stacktop -= n; /* pop id's */
       }
    }
    new_facet_id = ordinal(f_id) + 1;
    *(++stacktop) = (REAL)new_facet_id;
    *recalc_flag = 1;
    break;

    case CREATE_BODY_NODE:
       { body_id b_id = new_body();
          new_body_id = ordinal(b_id) + 1;
          set_original(b_id,new_body_id);
          *(++stacktop) = (REAL)new_body_id;
       }
       *recalc_flag = 1;
       break;


    case MERGE_VERTEX_NODE:
      { vertex_id v_id2; 
        vertex_id v_id1; 
        int v2 = (int)(*stacktop--);
        int v1 = (int)(*stacktop--);
        v_id2 = get_ordinal_id(VERTEX,v2-1); 
        v_id1 = get_ordinal_id(VERTEX,v1-1); 
        if ( (v1==0) || !valid_id(v_id1) )
        { sprintf(errmsg,"Invalid first vertex %d in vertex_merge.\n",v1);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(3886,errmsg,RECOVERABLE);
        }
        if ( (v2==0) || !valid_id(v_id2) )
        { sprintf(errmsg,"Invalid second vertex %d in vertex_merge.\n",v2);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(3916,errmsg, RECOVERABLE);
        }
        merge_vertex(v_id1,v_id2);
        *recalc_flag = 1;
      }
      break;

    case MERGE_EDGE_NODE:
      { edge_id e_id2; 
        edge_id e_id1; 
        int e2 = (int)(*stacktop--);
        int e1 = (int)(*stacktop--);
        e_id2 = get_ordinal_id(EDGE,abs(e2)-1); 
        if ( e2 < 0 )  invert(e_id2);
        e_id1 = get_ordinal_id(EDGE,abs(e1)-1); 
        if ( e1 < 0 ) invert(e_id1);
        if ( (e1==0) ||!valid_id(e_id1) )
        { sprintf(errmsg,"Invalid first edge %d in edge_merge.\n",e1);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(3878,errmsg,RECOVERABLE);
        }
        if ( (e2==0) || !valid_id(e_id2) )
        { sprintf(errmsg,"Invalid second edge %d in edge_merge.\n",e2);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(3912,errmsg, RECOVERABLE);
        }
        merge_edge(e_id1,e_id2);
        *recalc_flag = 1;
      }
      break;

    case MERGE_FACET_NODE:
      { facet_id f_id2; 
        facet_id f_id1; 
        int f2 = (int)(*stacktop--);
        int f1 = (int)(*stacktop--);
        f_id2 = get_ordinal_id(FACET,abs(f2)-1); 
        if ( f2 < 0 ) invert(f_id2);
        f_id1 = get_ordinal_id(FACET,abs(f1)-1); 
        if ( f1 < 0 ) invert(f_id1);
        if ( (f1==0) || !valid_id(f_id1) )
        { sprintf(errmsg,"Invalid first facet %d in facet_merge.\n",f1);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(3880,errmsg,RECOVERABLE);
        }
        if ( (f2==0) || !valid_id(f_id2) )
        { sprintf(errmsg,"Invalid second facet %d in facet_merge.\n",f2);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(3879,errmsg, RECOVERABLE);
        }
        merge_facet(f_id1,f_id2);
        *recalc_flag = 1;
      }
      break;

    case GET_TORUS_PERIODS_NODE:
         j = (int)(*stacktop--);
         i = (int)(*stacktop--);
         if ( (j<1) || (j > SDIM))
         { sprintf(errmsg,
              "Torus_periods index %d; must be between 1 and %d.\n",j,SDIM);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2015,errmsg,RECOVERABLE);
         }
         if ((i<1) || (i>SDIM) )
         { sprintf(errmsg,
              "Torus_periods index %d; must be between 1 and %d.\n",i,SDIM);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2016,errmsg,RECOVERABLE);
         }
         *++stacktop = web.torus_period[i-1][j-1]; /* 1-based indexing */
         break;

    case GET_INVERSE_PERIODS_NODE:
         j = (int)(*stacktop--);
         i = (int)(*stacktop--);
         if ( (j<1) || (j > SDIM) )
         { sprintf(errmsg,
              "Inverse_periods index %d; must be between 1 and %d.\n",j,SDIM);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2017,errmsg,RECOVERABLE);
         }
         if ( (i<1) || (i>SDIM) )
         { sprintf(errmsg,
              "Inverse_periods index %d; must be between 1 and %d.\n",i,SDIM);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2018,errmsg,RECOVERABLE);
         }
         *++stacktop = web.inverse_periods[i-1][j-1]; /* 1-based indexing */
         break;

    case SET_GAP_CONSTANT_NODE:
       web.spring_constant = *(stacktop--); 
       if ( everything_quantities_flag )
               GEN_QUANT(gap_quantity_num)->modulus = web.spring_constant;
       if ( web.spring_constant == 0.0 ) web.convex_flag = 0;
       else web.convex_flag = 1;
       *recalc_flag = 1;
       break;

    case GET_INTERNAL_NODE:
       *++stacktop = get_internal_variable(node->op1.name_id);
          break;
  
    case SET_INTERNAL_NODE:
    { REAL oldvalue=0.0;
      REAL val; 
      /* get old value */
      switch(node->op1.name_id)
      {
#ifdef MPI_EVOLVER
        case V_CORONA_STATE : oldvalue = mpi_corona_state; break;
#endif
        case V_BOUNDING_BOX_COLOR: oldvalue = bounding_box_color; break;
        case V_DETORUS_EPSILON: oldvalue = dt_eps; break;
        case V_PS_LABELSIZE:  oldvalue = ps_labelsize; break;
        case V_PS_STRINGWIDTH: oldvalue = ps_stringwidth; break;
        case V_PS_FIXEDEDGEWIDTH: oldvalue = ps_fixededgewidth; break;
        case V_PS_TRIPLEEDGEWIDTH: oldvalue = ps_tripleedgewidth; break;
        case V_PS_CONEDGEWIDTH: oldvalue = ps_conedgewidth; break;
        case V_PS_BAREEDGEWIDTH: oldvalue = ps_bareedgewidth; break;
        case V_PS_GRIDEDGEWIDTH: oldvalue = ps_gridedgewidth; break;
        case V_TOLERANCE: oldvalue = web.tolerance; break;
        case V_LAST_ERROR: oldvalue = last_error; break;
        case V_BRIGHTNESS: oldvalue = brightness; break;
        case V_BACKGROUND: oldvalue = background_color; break;
        case V_TARGET_TOLERANCE: oldvalue = web.target_tolerance;
              break;
        case V_HESS_EPSILON: oldvalue = hessian_epsilon; break;
        case GRAV_CONST_NODE: oldvalue = web.grav_const; break;
        case V_THICKNESS: oldvalue = thickness; break;
        case V_HESSIAN_SLANT_CUTOFF: oldvalue = hessian_slant_cutoff; break;
        case V_AMBIENT_PRESSURE: oldvalue = web.pressure; break;
        case V_DIFFUSION: oldvalue = web.diffusion_const;
            web.diffusion_flag = 1; break;
        case V_SCALE_SCALE: oldvalue = web.scale_scale; break;
        case V_SCALE: oldvalue = web.scale; break;
        case V_SCALE_LIMIT: oldvalue = web.maxscale; break;
        case V_TIME: oldvalue = total_time; break;
        case V_CHECK_COUNT_NODE: oldvalue = check_count; break;
        case V_VISIBILITY_DEBUG_NODE: oldvalue = visdebuglevel; break;
        case V_SCROLLBUFFERSIZE_NODE: oldvalue = scrollbuffersize; break;
        case V_BREAKFLAG_NODE: oldvalue = breakflag; break;
        case V_STRING_CURVE_TOLERANCE: oldvalue = string_curve_tolerance;
             break;
        case V_MINDEG_DEBUG_LEVEL: oldvalue = mindeg_debug_level; break;
        case V_MINDEG_MARGIN: oldvalue = mindeg_margin; break;
        case V_MINDEG_MIN_REGION_SIZE: oldvalue = mindeg_min_region_size; break;
        case V_WINDOW_ASPECT_RATIO: oldvalue = window_aspect_ratio; break;
        case V_PICKVNUM: oldvalue = pickvnum; break;
        case V_PICKENUM: oldvalue = pickenum; break;
        case V_PICKFNUM: oldvalue = pickfnum; break;
        case V_JIG_TEMP: oldvalue = web.temperature; break;
        case V_LINEAR_METRIC_MIX: oldvalue = linear_metric_mix; break;
        case V_GAP_CONSTANT: oldvalue = web.spring_constant; break;
        case V_QUADRATIC_METRIC_MIX: 
              oldvalue = quadratic_metric_mix; break;
        case V_RANDOM_SEED: oldvalue = random_seed;break;
        case V_RANDOM: 
             sprintf(errmsg,"Cannot set random. Set random_seed instead.\n");
             stacktop--;
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2019,errmsg, WARNING);
             break;
        case V_INTEGRAL_ORDER: oldvalue = web.gauss2D_order; break;
        case V_INTEGRAL_ORDER_1D: oldvalue = web.gauss1D_order; break;
        case V_INTEGRAL_ORDER_2D: oldvalue = web.gauss2D_order; break;
        case V_AUTOCHOP_LENGTH: oldvalue = autochop_length; break;
        case GRAV_CONST_TOK: oldvalue = web.grav_const; break;
        default: 
             sprintf(errmsg,"Internal: illegal variable number %d.\n",
                 node->op1.name_id);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1209,errmsg, RECOVERABLE);
      }
      val = *(stacktop--);
      switch ( node->op2.assigntype )
      {    case ASSIGN_OP:    break;
           case PLUSASSIGN_OP: val +=  oldvalue; break;
           case SUBASSIGN_OP: val = oldvalue - val; break;
           case MULTASSIGN_OP: val = oldvalue * val; break;
           case DIVASSIGN_OP: val = oldvalue / val; break;
      }
      switch(node->op1.name_id)
      {    case GRAV_CONST_TOK: web.grav_const = val;
                if ( web.grav_const != 0.0 )
                  web.gravflag = 1;
                else
                  web.gravflag = 0;
                if (gravity_quantity_num >= 0 )
                   GEN_QUANT(gravity_quantity_num)->modulus =
                              web.gravflag ? web.grav_const : 0.0;
                *recalc_flag = 1;
                 break;
           case V_TOLERANCE: web.tolerance = val; break;
           case V_PS_LABELSIZE: ps_labelsize = val; break;
           case V_PS_STRINGWIDTH: ps_stringwidth = val; break;
           case V_PS_FIXEDEDGEWIDTH: ps_fixededgewidth = val; break;
           case V_PS_TRIPLEEDGEWIDTH: ps_tripleedgewidth = val; break;
           case V_PS_CONEDGEWIDTH: ps_conedgewidth = val; break;
           case V_PS_BAREEDGEWIDTH: ps_bareedgewidth = val; break;
           case V_PS_GRIDEDGEWIDTH: ps_gridedgewidth = val; break;
#ifdef MPI_EVOLVER
           case V_CORONA_STATE: mpi_set_corona((int)val); break;
#else
           case V_CORONA_STATE: ; break;
         
#endif
           case V_BRIGHTNESS: 
             if ( (val < 0.0) || (val > 1.0) )
             { sprintf(errmsg,"Brightness is %f; must be between 0 and 1.\n",
                    (DOUBLE)val);
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
               kb_error(2020,errmsg, RECOVERABLE);
             }
             else { brightness = val; update_display(); }
                 break;
           case V_BACKGROUND:
             background_color = (int)val; 
             update_display();
             break;

           case V_TARGET_TOLERANCE: web.target_tolerance = val;
              break;
           case V_HESS_EPSILON: hessian_epsilon = val;
             if ( hessian_epsilon < 0.0 )
             { sprintf(errmsg,"hessian_epsilon is negative!\n");
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                   file_names[node->file_no],node->line_no);
               kb_error(2558,errmsg, WARNING);
             }
             break;

           case GRAV_CONST_NODE: web.grav_const = val; 
             if ( web.grav_const != 0.0 )
               web.gravflag = 1;
             else
               web.gravflag = 0;
             if (gravity_quantity_num >= 0 )
               GEN_QUANT(gravity_quantity_num)->modulus =
                              web.gravflag ? web.grav_const : 0.0;
             break;

           case V_DIFFUSION: web.diffusion_const = val; break;
           case V_AMBIENT_PRESSURE: web.pressure = val; break;
           case V_HESSIAN_SLANT_CUTOFF: hessian_slant_cutoff = val; break;
           case V_THICKNESS: thickness = val; user_thickness_flag = 1;
                  update_display();break;
           case V_SCALE_SCALE: web.scale_scale = val; break;           
           case V_SCALE: web.scale = val; break;
           case V_SCALE_LIMIT: web.maxscale = val; break;
           case V_LAST_ERROR: last_error = (int)val; break;
           case V_TIME: total_time = val; break;
           case V_CHECK_COUNT_NODE: check_count = (int)val; break;
           case V_VISIBILITY_DEBUG_NODE: visdebuglevel = (int)val; break;
           case V_SCROLLBUFFERSIZE_NODE: scrollbuffersize = (int)val;
                set_scroll_size(scrollbuffersize); break;
           case V_BREAKFLAG_NODE: breakflag = (int)val; break;
           case V_STRING_CURVE_TOLERANCE: string_curve_tolerance = val; 
                update_display();break;
           case V_MINDEG_DEBUG_LEVEL: mindeg_debug_level = (int)val; break;
           case V_MINDEG_MARGIN: mindeg_margin = (int)val; break;
           case V_MINDEG_MIN_REGION_SIZE: mindeg_min_region_size = (int)val; break;
           case V_WINDOW_ASPECT_RATIO: 
             window_aspect_ratio = (REAL)val;
             if ( window_aspect_ratio <= 1.0 ) 
             { minclipx = -1.5/window_aspect_ratio; 
               maxclipx = 1.5/window_aspect_ratio;
               minclipy = -1.5; 
               maxclipy = 1.5;
             }
             else
             { minclipx = -1.5; 
               maxclipx = 1.5;
               minclipy = -1.5*window_aspect_ratio; 
               maxclipy = 1.5*window_aspect_ratio;
             }
             update_display(); break;
           case V_PICKVNUM: pickvnum = (int)val; break;
           case V_PICKENUM: pickenum = (int)val; break;
           case V_PICKFNUM: pickfnum = (int)val; break;
           case V_JIG_TEMP: web.temperature = val; break;
           case V_DETORUS_EPSILON: dt_eps = val; break;
           case V_BOUNDING_BOX_COLOR: 
                 if ( ((int)val < BLACK) || ((int)val > WHITE) )
                 { sprintf(errmsg, "Illegal value %d for bounding_box_color.\n",(int)val);
                   kb_error(3037,errmsg,RECOVERABLE);
                 }
                 bounding_box_color = (int)val;
                 update_display(); 
                 break;
           case V_LINEAR_METRIC_MIX: linear_metric_mix=val; break;
           case V_GAP_CONSTANT: web.spring_constant = val; break;
           case V_AUTOCHOP_LENGTH: autochop_length = val; break;
           case V_QUADRATIC_METRIC_MIX: 
                 quadratic_metric_mix=val; break;
           case V_RANDOM_SEED: random_seed = (int)val;
                 srand(random_seed); srand48(random_seed); 
                 kb_initr(random_seed); break;
           case V_RANDOM: sprintf(errmsg,
                    "Cannot set random. Set random_seed instead.\n"); 
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2021,errmsg,WARNING);
                stacktop--; break;
           case V_INTEGRAL_ORDER: 
                 web.gauss1D_order = (int)val;
                 set_by_user_gauss_1D = web.gauss1D_order;
                 web.gauss2D_order = (int)val;
                 set_by_user_gauss_2D = web.gauss2D_order;
                 gauss_setup();
                 if ( web.modeltype == LAGRANGE )
                 { gauss_lagrange_setup(
                      (web.dimension==1)?1:web.dimension-1,
                          web.lagrange_order,web.gauss1D_order);
                   gauss_lagrange_setup(web.dimension,
                      web.lagrange_order,web.gauss2D_order);
                 }
                 *recalc_flag = 1;
                 break;
           case V_INTEGRAL_ORDER_1D: 
                 web.gauss1D_order = (int)val;
                 set_by_user_gauss_1D = web.gauss1D_order;
                 if ( web.modeltype == LAGRANGE )
                     gauss_lagrange_setup(
                      (web.dimension==1)?1:web.dimension-1,
                          web.lagrange_order,web.gauss1D_order);
                 gauss_setup();
                 *recalc_flag = 1;
                 break;
           case V_INTEGRAL_ORDER_2D: 
                 web.gauss2D_order = (int)val;
                 set_by_user_gauss_2D = web.gauss2D_order;
                 gauss_setup();
                 if ( web.modeltype == LAGRANGE )
                   gauss_lagrange_setup(web.dimension,
                      web.lagrange_order,web.gauss2D_order);
                 *recalc_flag = 1;
                 break;
           default: 
               sprintf(errmsg,"Internal: illegal internal variable number %d.\n",
                  node->op1.name_id);
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
               kb_error(1114,errmsg,RECOVERABLE);
   
      }
      break;
    } /* end SET_INTERNAL_ */

    case SIZEOF_ATTR_NODE:
      *++stacktop = (REAL)EXTRAS(node->op2.eltype)[node->op1.extranum].array_spec.datacount;
       break;

    case SIZEOF_ARRAY_NODE:
      *++stacktop = (REAL)get_name_arrayptr(node->op1.name_id,localstack,localbase)->datacount;
       break;

    case SIZEOF_STRING_NODE:
     { char *s = *(char**)stacktop;
       *stacktop = s ? (REAL)strlen(s) : 0;
       if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
         myfree(s);
       break;
     }

    case TORDUP_NODE:
       i = (int)*(stacktop--);
       if ( ! web.torus_flag )
       { outstring("Torus model not in effect.\n");
         break;
       }
       if ( i < 1 || i > SDIM ) outstring("Improper period number.\n");
       else tordup(i-1);
       recalc();
       break;
     
    case SKINNY_NODE:
       sprintf(msg,"New edges: %d\n",web.facet_refine_count = skinny(*(stacktop--)));
       outstring(msg);
       recalc();
       break;

    case PUSHQPRESSURE_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->pressure;
          break;
    case PUSHQTARGET_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->target;
          break;
    case PUSHQMODULUS_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->modulus;
          break;
    case PUSHQTOLERANCE_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->tolerance;
          break;
    case PUSHMMODULUS_NODE:
     *++stacktop = METH_INSTANCE(node->op1.meth_id)->modulus;
          break;
    case PUSHQVALUE_NODE:
      { struct gen_quant *q = GEN_QUANT(node->op1.quant_id);
        if ( q->timestamp<global_timestamp )
                   calc_quants(q->flags&(Q_INFO|Q_ENERGY|Q_FIXED));
        *++stacktop = q->value;
      }
      break;
    case PUSHQFIXED_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->flags & Q_FIXED ? 1:0 ;
          break;
    case PUSHQENERGY_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->flags & Q_ENERGY ? 1:0 ;
          break;
    case PUSHQINFO_ONLY_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->flags & Q_INFO ? 1:0 ;
          break;
    case PUSHQCONSERVED_NODE:
     *++stacktop = GEN_QUANT(node->op1.quant_id)->flags & Q_CONSERVED ? 1:0 ;
          break;

    case PUSHMVALUE_NODE:
      { struct method_instance *q 
              = METH_INSTANCE(node->op1.meth_id);
        if ((q->timestamp<graph_timestamp) || (q->timestamp<web_timestamp))
                   calc_quants((Q_INFO|Q_ENERGY|Q_FIXED));
        *++stacktop = q->value;
      }
      break;

    case PUSHQVOLCONST_NODE:
      *++stacktop = GEN_QUANT(node->op1.quant_id)->volconst;
      break;
  
    case PUSH_NAMED_QUANTITY_NODE:
    case PUSH_METHOD_INSTANCE_NODE:
       *++stacktop = (REAL)node->op1.quant_id;
       break;
    
    case FIX_QUANTITY_NODE: case SET_Q_FIXED_NODE:
    { struct gen_quant *q = GEN_QUANT(node->op1.quant_id);
      q->flags &= ~(Q_ENERGY|Q_INFO|Q_CONSERVED);
      q->flags |= Q_FIXED;
      if ( everything_quantities_flag )
        if ( valid_id(q->b_id) )
          set_attr(q->b_id,FIXEDVOL);
    }
    break;

    case UNFIX_QUANTITY_NODE:
    { struct gen_quant *q = GEN_QUANT(node->op1.quant_id);
      q->flags &= ~(Q_ENERGY|Q_FIXED|Q_CONSERVED);
      q->flags |= Q_INFO;
      if ( everything_quantities_flag )
        if ( valid_id(q->b_id) )
          unset_attr(q->b_id,FIXEDVOL);
    }
    break;

    case SET_Q_INFO_NODE:
    { struct gen_quant *q = GEN_QUANT(node->op1.quant_id);
      q->flags &= ~(Q_ENERGY|Q_FIXED|Q_CONSERVED);
      q->flags |= Q_INFO;
      if ( everything_quantities_flag )
        if ( valid_id(q->b_id) )
          unset_attr(q->b_id,FIXEDVOL);
    }
    break;

    case SET_Q_ENERGY_NODE:
    { struct gen_quant *q = GEN_QUANT(node->op1.quant_id);
      if ( everything_quantities_flag )
        if ( valid_id(q->b_id) )
        { sprintf(errmsg,"Cannot set body volume to ENERGY.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2416,errmsg,RECOVERABLE);
        }
      q->flags &= ~(Q_FIXED|Q_INFO|Q_CONSERVED);
      q->flags |= Q_ENERGY;
    }
    break;

    case SET_Q_CONSERVED_NODE:
    { struct gen_quant *q = GEN_QUANT(node->op1.quant_id);
      if ( everything_quantities_flag )
        if ( valid_id(q->b_id) )
        { sprintf(errmsg,"Cannot set body volume to CONSERVED.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2546,errmsg,RECOVERABLE);
        }
      q->flags &= ~(Q_FIXED|Q_INFO|Q_ENERGY);
      q->flags |= Q_CONSERVED;
    }
    break;


    case UNFIX_PARAMETER_NODE: 
    { struct global *p = globals(node->op1.name_id);
      int old = p->flags & OPTIMIZING_PARAMETER;
      if ( !old )
      { if ( p->type != REAL_TYPE )
        { sprintf(errmsg,"Cannot unfix \"%s\" since it is not type REAL.\n",
            p->name);
          kb_error(4124,errmsg,RECOVERABLE);
        }
        if ( optparamcount >= MAXOPTPARAM-1 )
        { sprintf(errmsg,"Too many optimizing parameters.  Increase MAXOPTPARAM in extern.h and recompile.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2022,errmsg,RECOVERABLE);
        }
        p->flags |= OPTIMIZING_PARAMETER;
        memset((char*)&optparam[optparamcount],0,sizeof(struct optparam_t));
        optparam[optparamcount++].pnum = node->op1.name_id;
        if ( !everything_quantities_flag ) convert_to_quantities();
        sprintf(msg,"%s now optimizing parameter. (was not)\n",p->name);
      }
      else sprintf(msg,"%s now optimizing parameter. (already was)\n",p->name);
      break;
    }

    case FIX_PARAMETER_NODE:
    { struct global *p = globals(node->op1.name_id);
      int old = p->flags & OPTIMIZING_PARAMETER;
      int spot;
      if ( old )
      { for ( spot = 0 ; spot < optparamcount ; spot++ )
           if ( optparam[spot].pnum == node->op1.name_id ) break;
        if ( spot == optparamcount )
        { sprintf(errmsg,"`%s` not found in optimizing parameter list.\n",p->name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2023,errmsg,WARNING);
          break;
        }
        p->flags &= ~OPTIMIZING_PARAMETER;
        optparam[spot] = optparam[--optparamcount];
        sprintf(msg,"%s now nonoptimizing parameter. (was not)\n",p->name);
      }
      else sprintf(msg,"%s now nonoptimizing parameter. (already was)\n",p->name);
      break;
    }

    /* attribute setting in set statements */
    case SET_ATTRIBUTE_NODE: 
    case SET_ATTRIBUTE_L_NODE: 
         if ( node->op1.localnum ) 
              id = *(element_id*)get_localp(node->op1.localnum);
         else id = q_id;
         parallel_update_flag[id_type(id)] = 1; /* set when element info changed */
         switch ( node->op2.attr_kind )
         {
           case SET_FIXED_NODE: set_attr(id,FIXED); break;
           case SET_TRIPLE_PT_NODE: set_attr(id,TRIPLE_PT); --stacktop; break;
           case SET_TETRA_PT_NODE: set_attr(id,TETRA_PT); --stacktop; break;
           case SET_AXIAL_POINT_NODE: set_attr(id,AXIAL_POINT); --stacktop; break;
   
           case SET_EXTRA_ATTR_NODE:
            { struct extra *ext;
              int spot; /* index */
              n = node->op3.extra_info & ((1<<ESHIFT)-1);
              if ( node->op1.localnum ) 
                 id = *(element_id*)get_localp(node->op1.localnum);
              else id = q_id;
              ext = EXTRAS(id_type(id)) + n;
              /* get index */
              spot = 0;
              for ( k = 0 ; k < ext->array_spec.dim ; k++ )
              { int j = (int)(stacktop[-ext->array_spec.dim+k+1]);
                spot *= ext->array_spec.sizes[k];
                if ( j > ext->array_spec.sizes[k] )
                { sprintf(errmsg,
                     "Attribute %s index %d is %d; maximum is %d (in %s).\n",
                     ext->name,k+1,j,ext->array_spec.sizes[k],ext->name);
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                  kb_error(1210,errmsg,RECOVERABLE);
                }
                if ( j < 1 )
                { sprintf(errmsg,
                     "Attribute %s index %d is %d; must be positive.\n",
                        ext->name,k,j); 
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                  kb_error(1129,errmsg,RECOVERABLE);
                }
                spot += (int)(stacktop[-ext->array_spec.dim+k+1]) - 1;
              }
              stacktop -= ext->array_spec.dim;
   
              switch ( ext->type )
              { case REAL_TYPE:  ((REAL*)get_extra(id,n))[spot] = *(stacktop--); 
                   break;
                case INTEGER_TYPE:
                   ((int*)get_extra(id,n))[spot] = (int)*(stacktop--);
                   break;
                case ULONG_TYPE:
                   ((unsigned long*)get_extra(id,n))[spot] = 
                     (unsigned long)*(stacktop--);
                   break;
                case USHORT_TYPE:
                   ((unsigned short*)get_extra(id,n))[spot] = 
                     (unsigned short)*(stacktop--);
                   break;
                case UINT_TYPE:
                   ((unsigned int*)get_extra(id,n))[spot] = 
                     (unsigned int)*(stacktop--);
                   break;
                case UCHAR_TYPE:
                   ((unsigned char*)get_extra(id,n))[spot] = 
                     (unsigned char)*(stacktop--);
                   break;
                case LONG_TYPE:
                   ((long int*)get_extra(id,n))[spot] = 
                     (long int)*(stacktop--);
                   break;
                case SHORT_TYPE:
                   ((short*)get_extra(id,n))[spot] = 
                     (short)*(stacktop--);
                   break;
                case CHAR_TYPE:
                   ((char*)get_extra(id,n))[spot] = 
                     (char)*(stacktop--);
                   break;
                case ELEMENTID_TYPE:
                case VERTEX_TYPE:
                case EDGE_TYPE:
                case FACET_TYPE:
                case BODY_TYPE:
                case FACETEDGE_TYPE:
                   ((element_id*)get_extra(id,n))[spot] = 
                     *(element_id*)(stacktop--);
                   break;
                case PTR_TYPE:
                   ((char**)get_extra(id,n))[spot] = 
                     *(char**)(stacktop--);
                   break;
                case BOUNDARY_TYPE:
                case CONSTRAINT_TYPE:
                case QUANTITY_TYPE:
                case INSTANCE_TYPE:
                case PROCEDURE_TYPE:
                   ((int*)get_extra(id,n))[spot] = (int)*(stacktop--);
                   break;
              }
           }
           break;
   
          case SET_DENSITY_NODE:
          { REAL density = *(stacktop--);
              if ( density != 1.0 ) set_attr(id,DENSITY);
              switch ( id_type(id) )
                 { case EDGE: set_edge_density(id,density); 
                       pressure_set_flag = 0; break;
                   case FACET: set_facet_density(id,density);
                       pressure_set_flag = 0; break;
                   case BODY: set_body_density(id,density);
                       web.gravflag = 1;
                       if (gravity_quantity_num >= 0 )
                           GEN_QUANT(gravity_quantity_num)->modulus =
                                 web.grav_const;
                       break;
                 }
              *recalc_flag = 1;
             }
           break;
   
      case SET_PHASE_NODE:
          { int phase = (int)*(stacktop--);
              switch ( id_type(id) )
                 { case BODY: set_b_phase(id,phase); break;
                   case FACET: set_f_phase(id,phase); break;
                 }
              *recalc_flag = 1;
           }
           break;
   
         case SET_VOLCONST_NODE: 
           { REAL v = *(stacktop--);
             REAL oldvcon = get_body_volconst(id);
             set_body_volconst(id,v); 
             set_body_volume(id,get_body_volume(id)+v-oldvcon,SETSTAMP);
             set_body_oldvolume(id,get_body_oldvolume(id)+v-oldvcon);
             if ( everything_quantities_flag )
               GEN_QUANT(get_body_volquant(id))->volconst = v;
             *recalc_flag = 1;
             break;
           }
   
         case SET_TARGET_NODE: 
           { REAL v = *(stacktop--);
             if ( get_attr(id) & PRESSURE )
             { sprintf(errmsg,"Must unset body pressure before fixing target.\n");
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
               kb_error(2024,errmsg, RECOVERABLE);
             }
             set_attr(id,FIXEDVOL);
             set_body_fixvol(id,v);  /* takes care of quantity */
             reset_conj_grad();
             *recalc_flag = 1;
             break;
           }
   
           case SET_PRESSURE_NODE: 
           { REAL p = *(stacktop--);
             unset_attr(id,FIXEDVOL); 
             set_attr(id,PRESSURE);
             web.pressflag = 1;
             set_body_pressure(id,p); 
             reset_conj_grad();
             if ( everything_quantities_flag )
             { struct gen_quant *q = GEN_QUANT(get_body_volquant(id));
               if ( p == 0.0 )
               { q->modulus = 1.0;
                 if ( !(q->flags & Q_INFO) )
                 { q->flags &= ~(Q_ENERGY|Q_FIXED|Q_CONSERVED);
                   q->flags |= Q_INFO;
                 }
               }
               else 
               { q->modulus = -p;
                 if ( !(q->flags & Q_ENERGY ) )
                 { q->flags &= ~(Q_INFO|Q_FIXED|Q_CONSERVED);
                   q->flags |= Q_ENERGY;
                 }
               }
             }
             *recalc_flag = 1;
             break;
             }
   
           case SET_OPACITY_NODE:
             *(REAL*)(get_extra(id,opacity_attr)) = *(stacktop)--;
             *recalc_flag = 1;
             break;
   
          case SET_CONSTRAINT_NODE:
           { int con = (int)*(stacktop--);
             struct constraint *constr;
             constr = get_constraint(con); 
             if ( (con<0) || (con>=web.maxcon) || !(constr->attr & IN_USE))
               { sprintf(errmsg,"Illegal constraint number: %d.\n",con);
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(1212,errmsg,RECOVERABLE);
               }
             if ( get_attr(id) & BOUNDARY ) 
             { sprintf(errmsg,
                 "Cannot set constraint on %s %s since it is on a boundary.\n",
                  typenames[id_type(id)],ELNAME(id));
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
               kb_error(4111,errmsg,RECOVERABLE);
             }
              set_attr(id,CONSTRAINT);
              if ( constr->attr & CON_ENERGY )
                 set_attr(id, BDRY_ENERGY);
              if ( constr->attr & CON_CONTENT )
                 set_attr(id, BDRY_CONTENT);
              switch ( id_type(id) )
                 { case VERTEX: set_v_constraint_map(id,con); 
                              project_v_constr(id,ACTUAL_MOVE,RESET_ONESIDEDNESS);
                              break;
                   case EDGE: set_e_constraint_map(id,con); break;
                   case FACET: set_f_constraint_map(id,con); break;
                 }
              *recalc_flag = 1;
             }
           break;
      
            case SET_BOUNDARY_NODE:
            { struct boundary *bdry;
              k =  (int)*(stacktop--);  /* boundary number */
              bdry = web.boundaries+abs(k);
              if (  (abs(k) >= web.bdrymax) || !(bdry->attr&IN_USE) ) 
              { sprintf(errmsg,"Boundary %d is not valid.\n",k);
                kb_error(3374,errmsg,RECOVERABLE);
              }
              if ( get_attr(q_id) & CONSTRAINT )
              { sprintf(errmsg,"Cannot put %s %s on a boundary since it is already on a constraint.\n",
                typenames[id_type(q_id)],ELNAME(q_id));
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                   file_names[node->file_no],node->line_no);
                kb_error(6199,errmsg,RECOVERABLE);
              }

              if ( get_attr(q_id) & BOUNDARY )
              { struct boundary *qbdry;
                switch ( id_type(q_id) )
                { case VERTEX: qbdry = get_boundary(q_id); break;
                  case EDGE:   qbdry = get_edge_boundary(q_id); break;
                  case FACET:  qbdry = get_facet_boundary(q_id); break;
                  default:     qbdry = NULL;  /* error message below */
                }
                if ( qbdry == bdry ) break;
                sprintf(errmsg,"%s %s already on a different boundary.\n",
                   typenames[id_type(q_id)],ELNAME(q_id));
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                kb_error(3375,errmsg,RECOVERABLE);
              }
              set_attr(q_id,BOUNDARY);
              if ( k < 0 )
                set_attr(q_id,NEGBOUNDARY);
       
              switch(id_type(q_id))
                { case VERTEX:
                  { REAL *x = get_coord(q_id);
                    int n;
                    set_boundary_num(q_id,abs(k));
                    PUSH_TRACE;
                    for ( n = 0 ; n < SDIM ; n++ )
                     if ( bdry->coordf[n]->root != NULL )
                      x[n] = eval(bdry->coordf[n],get_param(q_id),q_id,NULL);
                    POP_TRACE;
                    break;
                  }
                  case EDGE:
                   set_edge_boundary_num(q_id,abs(k));
                   break;
                  case FACET:
                   set_facet_boundary_num(q_id,abs(k));
                   break;
                  default: 
                   sprintf(errmsg,"Bad element type for boundary.\n");
                   sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                       file_names[node->file_no],node->line_no);
                   kb_error(3376,errmsg, RECOVERABLE);
                 }
                if ( bdry->attr & CON_ENERGY )
                   apply_method_num(q_id,bdry->energy_method);
                if ( bdry->attr & CON_CONTENT )
                { if ( (web.representation == STRING) && (id_type(q_id) == VERTEX) )
                    fixup_vertex_content_meths(q_id);
                  else if ( (web.representation == SOAPFILM) && (id_type(q_id) == EDGE) )
                    fixup_edge_content_meths(q_id);
                }  
              
                /* go back to next element generator */
                node += node->op1.skipsize - 1;  /* back to start of loop */
                break;
              }
      
           case SET_ORIGINAL_NODE:
             { int orig = (int)*(stacktop--);
               if ( orig == 0 )
                 set_original(id,NULLID);
               else if ( orig > 0 )
                 set_original(id,(orig-1) | ((element_id)id_type(id)<<TYPESHIFT) | 
                                                        VALIDMASK );
               else
               { sprintf(errmsg,"Cannot have negative \"original\" attribute.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(5987,errmsg,RECOVERABLE);
               }
             }
             break;
   
           case SET_COLOR_NODE:
           { int color = (int)*(stacktop--);
              switch ( id_type(id) )
                 { case EDGE: set_edge_color(id,color); break;
                   case FACET: set_facet_color(id,color); break;
                 }
              *update_display_flag = 1;
             }
           break;
   
           case SET_FRONTCOLOR_NODE:
           { int color = (int)*(stacktop--);
              set_facet_frontcolor(id,color);
              *update_display_flag = 1;
             }
             break;
   
          case SET_BACKCOLOR_NODE:
           { int color = (int)*(stacktop--);
              set_facet_backcolor(id,color);
              *update_display_flag = 1;
             }
             break;
   
     case SET_WRAP_NODE:
           { int w = (int)*(stacktop--);
             if ( web.symmetry_flag )
             { set_edge_wrap(id,w);
               *update_display_flag = 1;
             }
           }
           break;
   
          case SET_ORIENTATION_NODE:
           { int orient = (int)*(stacktop--);
             if ( orient < 0 ) 
               set_attr(id,NEGBOUNDARY);
             else unset_attr(id,NEGBOUNDARY);
           }
           break; 
   
          case SET_COORD_NODE: case SET_COORD_1_NODE: 
             if ( node->right ) /* indexed */
             { int dim = (int)*(stacktop--)-1;
               if ( (dim >= SDIM) || (dim < 0) )
                   { sprintf(errmsg,
                      "Trying to set coordinate %d, in space dimension %d.\n",
                           dim+1,SDIM);
                     sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                     kb_error(2026,errmsg,RECOVERABLE);
                   }
                 get_coord(id)[dim] = *(stacktop--);
                 *recalc_flag = 1;
                 break;
              }
              /* else fall through */
   
            case SET_COORD_2_NODE:
            case SET_COORD_3_NODE: case SET_COORD_4_NODE: case SET_COORD_5_NODE:
            case SET_COORD_6_NODE: case SET_COORD_7_NODE: case SET_COORD_8_NODE:
              get_coord(id)[node->op2.attr_kind-SET_COORD_1_NODE] = *(stacktop--);
              *recalc_flag = 1;
              break;
   
            case SET_PARAM_NODE: case SET_PARAM_1_NODE: case SET_PARAM_2_NODE:
            case SET_PARAM_3_NODE: case SET_PARAM_4_NODE: case SET_PARAM_5_NODE:
            case SET_PARAM_6_NODE: case SET_PARAM_7_NODE: case SET_PARAM_8_NODE:
              { int pnum;
                if ( node->right ) pnum = (int)*(stacktop--);
                else pnum = node->op2.attr_kind-SET_PARAM_1_NODE; 
                
                if ( get_vattr(id) & BOUNDARY )
                { struct boundary *boundary = get_boundary(id);
                  REAL *param = get_param(id);
                  REAL *xx = get_coord(id);
                  
                  if ( pnum > boundary->pcount )
                  { sprintf(errmsg,"Parameter number is %d; maximum is %d.\n",
                         pnum,boundary->pcount);
                    sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                         file_names[node->file_no],node->line_no);
                    kb_error(1149,errmsg,RECOVERABLE);
                  }
                  param[pnum] = *(stacktop--);
                  PUSH_TRACE;
                  for ( k = 0 ; k < SDIM ; k++ )
                     xx[k] = eval(boundary->coordf[k],param,id,NULL);
                  POP_TRACE;
                  *recalc_flag = 1;
               }
               else /* just parking a value probably */ 
               { REAL *param = get_param(id);
                 if ( pnum > web.maxparam )
                 { sprintf(errmsg,"Parameter index too high.\n");
                   sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                   kb_error(6531,errmsg,RECOVERABLE);
                 }
                 param[pnum] = *(stacktop--);
               }
              }
              break;
   
             case SET_FRONTBODY_NODE:
             case SET_BACKBODY_NODE:
               b_id = get_ordinal_id(BODY,(int)*(stacktop--)-1);
               if ( b_id && !valid_id(b_id) ) 
               { sprintf(errmsg,"Invalid body in SET FRONTBODY or BACKBODY.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2027,errmsg,RECOVERABLE);
               }
               set_body((node->op2.attr_kind==SET_FRONTBODY_NODE?id:inverse_id(id)),b_id);
               *recalc_flag = 1;
               break;
    
            default: 
               sprintf(errmsg,"Unhandled SET attribute %d\n",node->op2.attr_kind);
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
               kb_error(2028,errmsg,RECOVERABLE);
               break;
   
         } /* end SET_ATTRIBUTE */
         break;

    /*****************************
    * assignop attribute values  *
    * for single element assigns *
    *****************************/
    case SET_ATTRIBUTE_A_NODE: 
      { int assign_type = node[1].op1.assigntype;
        if ( node->op1.localnum ) 
          id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        parallel_update_flag[id_type(id)] = 1; /* set when element info changed */
        switch ( node->op2.attr_kind )
        {  /* Those attributes not fitted for arithmetic assignment 
              have error test for that in yexparse() */
   
           case SET_ORIENTATION_NODE:
             if ( *(stacktop--) >= 0.0 )
                 unset_attr(id,NEGBOUNDARY);
             else  set_attr(id,NEGBOUNDARY);
             break;
   
           case SET_EXTRA_ATTR_NODE:
           { struct extra *ext;
             int spot; /* index */
             n = node->op3.extra_info & ((1<<ESHIFT)-1);
             if ( node->op1.localnum ) 
               id = *(element_id*)get_localp(node->op1.localnum);
             else id = q_id;
             ext = EXTRAS(id_type(id)) + n;
              /* get index */
              spot = 0;
              for ( k = 0 ; k < ext->array_spec.dim ; k++ )
              { int j = (int)(stacktop[-ext->array_spec.dim+k+1]);
                spot *= ext->array_spec.sizes[k];
                if ( j > ext->array_spec.sizes[k] )
                { sprintf(errmsg,
                     "Attribute %s index %d is %d; maximum is %d (in %s).\n",
                     ext->name,k+1,j,ext->array_spec.sizes[k],ext->name);
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                  kb_error(2524,errmsg,RECOVERABLE);
                }
                if ( j < 1 )
                { sprintf(errmsg,"Attribute %s index %d is %d; must be positive.\n",
                        ext->name,k,j); 
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                  kb_error(3365,errmsg,RECOVERABLE);
                }
                spot += (int)(stacktop[-ext->array_spec.dim+k+1]) - 1;
              }
              stacktop -= ext->array_spec.dim;
   
   #define ATTRCASE(type) \
            switch ( assign_type ) \
            { case ASSIGN_OP: \
                ((type*)get_extra(id,n))[spot] = (type)*(stacktop--); break; \
              case PLUSASSIGN_OP: \
                ((type*)get_extra(id,n))[spot] += (type)*(stacktop--); break; \
              case SUBASSIGN_OP: \
                ((type*)get_extra(id,n))[spot] -= (type)*(stacktop--); break; \
              case MULTASSIGN_OP: \
                ((type*)get_extra(id,n))[spot] *= (type)*(stacktop--); break; \
              case DIVASSIGN_OP: \
                ((type*)get_extra(id,n))[spot] /= (type)*(stacktop--); break; \
            } 
   
             switch ( ext->type )
             { case REAL_TYPE:  
                   switch ( assign_type )
                   { case ASSIGN_OP:
                       ((REAL*)get_extra(id,n))[spot] = *(stacktop--); break;
                      case PLUSASSIGN_OP:
                       ((REAL*)get_extra(id,n))[spot] += *(stacktop--); break;
                      case SUBASSIGN_OP:
                       ((REAL*)get_extra(id,n))[spot] -= *(stacktop--); break;
                      case MULTASSIGN_OP:
                       ((REAL*)get_extra(id,n))[spot] *= *(stacktop--); break;
                      case DIVASSIGN_OP:
                       ((REAL*)get_extra(id,n))[spot] /= *(stacktop--); break;
                   }
                   break;
                 case INTEGER_TYPE:
                   switch ( assign_type )
                   { case ASSIGN_OP:
                       ((int*)get_extra(id,n))[spot] = (int)*(stacktop--); break;
                      case PLUSASSIGN_OP:
                       ((int*)get_extra(id,n))[spot] += (int)*(stacktop--); break;
                      case SUBASSIGN_OP:
                       ((int*)get_extra(id,n))[spot] -= (int)*(stacktop--); break;
                      case MULTASSIGN_OP:
                       ((int*)get_extra(id,n))[spot] = 
                          (int)(((int*)get_extra(id,n))[spot] * *(stacktop--));break;
                      case DIVASSIGN_OP:
                       ((int*)get_extra(id,n))[spot] = 
                          (int)(((int*)get_extra(id,n))[spot] / *(stacktop--));break;
                   }
                   break;
   
                 case ULONG_TYPE:
                   ATTRCASE(unsigned long int);
                   break;
   
                 case LONG_TYPE:
                   ATTRCASE(long int);
                   break;
   
                 case UINT_TYPE:
                   ATTRCASE(unsigned int);
                   break;
   
                 case UCHAR_TYPE:
                   ATTRCASE(unsigned char);
                   break;
   
                 case CHAR_TYPE:
                   ATTRCASE(char);
                   break;
   
                 case SHORT_TYPE:
                   ATTRCASE(short int);
                   break;
   
                 case USHORT_TYPE:
                   ATTRCASE(unsigned short int);
                   break;
   
                 case ELEMENTID_TYPE:
                 case VERTEX_TYPE:
                 case EDGE_TYPE:
                 case FACET_TYPE:
                 case BODY_TYPE:
                 case FACETEDGE_TYPE:
                   switch ( assign_type )
                   { case ASSIGN_OP:
                       ((element_id*)get_extra(id,n))[spot] = 
                           *(element_id*)(stacktop--); break;
                     default: 
                         { sprintf(errmsg,"Illegal assignment operation on element id attribute.\n");
                           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
                            kb_error(3453, errmsg,RECOVERABLE);
                         }
                   }
   
                 case PTR_TYPE:
                   switch ( assign_type )
                   { case ASSIGN_OP:
                       ((char**)get_extra(id,n))[spot] = 
                           *(char**)(stacktop--); break;
                     default: 
                         { sprintf(errmsg,"Illegal assignment operation on pointer attribute.\n");
                           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
                           kb_error(3454, errmsg, RECOVERABLE);
                         }
                   }
                 case BOUNDARY_TYPE:
                 case CONSTRAINT_TYPE:
                 case QUANTITY_TYPE:
                 case INSTANCE_TYPE:
                 case PROCEDURE_TYPE:
                   switch ( assign_type )
                   { case ASSIGN_OP:
                       ((int*)get_extra(id,n))[spot] = 
                           *(int*)(stacktop--); break;
                     default: 
                         { sprintf(errmsg,"Illegal assignment operation on attribute.\n");
                           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
                           kb_error(3682,errmsg,RECOVERABLE);
                         }
                   }
              }
           }
           break;

           case SET_DENSITY_NODE:
           { REAL density = *(stacktop--);
              REAL oldvalue = 0.0;
              if ( density != 1.0 ) set_attr(id,DENSITY);
              switch ( id_type(id) )
                 { case EDGE: oldvalue = get_edge_density(id); break;
                   case FACET: oldvalue = get_facet_density(id); break;
                   case BODY: oldvalue = get_body_density(id); break;
                 }
              switch ( assign_type )
                 { case ASSIGN_OP: break;
                   case PLUSASSIGN_OP: density += oldvalue; break;
                   case SUBASSIGN_OP: density = oldvalue - density; break;
                   case MULTASSIGN_OP: density *= oldvalue; break;
                   case DIVASSIGN_OP: density = oldvalue / density; break;
                 }
              switch ( id_type(id) )
                 { case EDGE: set_edge_density(id,density); 
                       pressure_set_flag = 0; break;
                   case FACET: set_facet_density(id,density);
                       pressure_set_flag = 0; break;
                   case BODY: set_body_density(id,density);
                       web.gravflag = 1;
                       if (gravity_quantity_num >= 0 )
                           GEN_QUANT(gravity_quantity_num)->modulus =
                                 web.grav_const;
                       break;
                 }
              *recalc_flag = 1;
             }
           break;
   
           case SET_PHASE_NODE:
           { int phase = (int)*(stacktop--);
              switch ( id_type(id) )
                 { case BODY: set_b_phase(id,phase); break;
                   case FACET: set_f_phase(id,phase); break;
                 }
              *recalc_flag = 1;
           }
           break;
   
           case SET_VOLCONST_NODE: 
           { REAL v = *(stacktop--);
              REAL oldvalue = get_body_volconst(id); 
              switch ( assign_type )
                 { case ASSIGN_OP:    break;
                   case PLUSASSIGN_OP: v +=  oldvalue; break;
                   case SUBASSIGN_OP: v = oldvalue - v; break;
                   case MULTASSIGN_OP: v = oldvalue * v; break;
                   case DIVASSIGN_OP: v = oldvalue / v; break;
                 }
             set_body_volconst(id,v); 
             set_body_volume(id,get_body_volume(id)+v-oldvalue,SETSTAMP);
             set_body_oldvolume(id,get_body_oldvolume(id)+v-oldvalue);
             if ( everything_quantities_flag )
               GEN_QUANT(get_body_volquant(id))->volconst = v;
              *recalc_flag = 1;
             break;
           }

           case SET_TARGET_NODE: 
           { REAL v = *(stacktop--);
              REAL oldvalue = get_body_fixvol(id); 
              switch ( assign_type )
                 { case ASSIGN_OP:    break;
                   case PLUSASSIGN_OP: v +=  oldvalue; break;
                   case SUBASSIGN_OP: v = oldvalue - v; break;
                   case MULTASSIGN_OP: v = oldvalue * v; break;
                   case DIVASSIGN_OP: v = oldvalue / v; break;
                 }
              if ( get_attr(id) & PRESSURE )
              { sprintf(errmsg,"Must unset body pressure before fixing target.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2029,errmsg,RECOVERABLE);
              }
             set_attr(id,FIXEDVOL);
             set_body_fixvol(id,v); 
             *recalc_flag = 1;
             break;
           }
           case SET_PRESSURE_NODE: 
           { REAL p = *(stacktop--);
              REAL oldvalue = get_body_pressure(id); 
              switch ( assign_type )
                 { case ASSIGN_OP:    break;
                   case PLUSASSIGN_OP: p +=  oldvalue; break;
                   case SUBASSIGN_OP: p = oldvalue - p; break;
                   case MULTASSIGN_OP: p = oldvalue * p; break;
                   case DIVASSIGN_OP: p = oldvalue / p; break;
                 }
              if ( get_attr(id) & FIXEDVOL )
              { sprintf(errmsg,"Must unset body target before fixing pressure.\n");
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                kb_error(2030,errmsg,RECOVERABLE);
              }
             set_attr(id,PRESSURE);
             set_body_pressure(id,p);    web.pressflag = 1;
             if ( everything_quantities_flag )
             { struct gen_quant *q = GEN_QUANT(get_body_volquant(id));
               if ( p == 0.0 )
               { q->modulus = 1.0;
                  if ( !(q->flags & Q_INFO) )
                  { q->flags &= ~(Q_ENERGY|Q_FIXED|Q_CONSERVED);
                     q->flags |= Q_INFO;
                  }
               }
               else 
               { q->modulus = -p;
                  if ( !(q->flags & Q_ENERGY ) )
                  { q->flags &= ~(Q_INFO|Q_FIXED|Q_CONSERVED);
                     q->flags |= Q_ENERGY;
                  }
               }
             }
              *recalc_flag = 1;
             break;
           }

           case SET_OPACITY_NODE:
             facet_alpha = *(stacktop)--;
             break;
   
           case SET_CONSTRAINT_NODE:
           { int con = (int)*(stacktop--);
              struct constraint *constr = get_constraint(con);
              if ( (con<0) || (con>=web.maxcon) || !(constr->attr & IN_USE))
              { sprintf(errmsg,"Illegal constraint number: %d.\n",con);
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                kb_error(1146,errmsg,RECOVERABLE);
              } 
             if ( get_attr(id) & BOUNDARY ) 
             { sprintf(errmsg,
                 "Cannot set constraint on %s %s since it is on a boundary.\n",
                  typenames[id_type(id)],ELNAME(id));
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
               kb_error(5111,errmsg,RECOVERABLE);
             }

              set_attr(id,CONSTRAINT);
              if ( constr->attr & CON_ENERGY )
                 set_attr(id, BDRY_ENERGY);
              if ( constr->attr & CON_CONTENT )
                 set_attr(id, BDRY_CONTENT);
              switch ( id_type(id) )
                 { case VERTEX: set_v_constraint_map(id,con); 
                              project_v_constr(id,ACTUAL_MOVE,RESET_ONESIDEDNESS);
                              break;
                   case EDGE: set_e_constraint_map(id,con); break;
                   case FACET: set_f_constraint_map(id,con); break;
                 }
              *recalc_flag = 1;
             }
           break;
   
           case SET_ORIGINAL_NODE:
             set_original(id,(int)*(stacktop--)-1 | ((element_id)id_type(id)<<TYPESHIFT)
                  | VALIDMASK);
             break;
   
           case SET_COLOR_NODE:
           { int color = (int)*(stacktop--);
              switch ( id_type(id) )
                 { case EDGE: set_edge_color(id,color); break;
                   case FACET: set_facet_color(id,color); break;
                 }
              *update_display_flag = 1;
           }
           break;
   
           case SET_FRONTCOLOR_NODE:
           { int color = (int)*(stacktop--);
              set_facet_frontcolor(id,color);
              *update_display_flag = 1;
           }
           break;
   
           case SET_BACKCOLOR_NODE:
           { int color = (int)*(stacktop--);
              set_facet_backcolor(id,color);
              *update_display_flag = 1;
           }
           break;
   
           case SET_WRAP_NODE:
           { int w = (int)*(stacktop--);
             if ( web.symmetry_flag )
             { set_edge_wrap(id,w);
               *update_display_flag = 1;
             }
           }
           break;
   
           case SET_COORD_NODE: case SET_COORD_1_NODE: 
             if ( node->right ) /* indexed */
             { int dim = (int)*(stacktop--)-1;
               REAL oldvalue = get_coord(id)[dim]; 
               REAL v;
               if ( (dim >= SDIM) || (dim < 0) )
                   { sprintf(errmsg,
                      "Trying to set coordinate %d, in space dimension %d.\n",
                           dim,SDIM);
                     sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                     kb_error(2031,errmsg,RECOVERABLE);
                   }
               v = *(stacktop--);
               switch ( assign_type )
                 { case ASSIGN_OP:  break;
                   case PLUSASSIGN_OP: v +=  oldvalue; break;
                   case SUBASSIGN_OP: v = oldvalue - v; break;
                   case MULTASSIGN_OP: v = oldvalue * v; break;
                   case DIVASSIGN_OP: v = oldvalue / v; break;
                 }
                 get_coord(id)[dim] = v;
                 *recalc_flag = 1;
                 break;
              }
              /* else fall through */
   
          case SET_COORD_2_NODE:
          case SET_COORD_3_NODE: case SET_COORD_4_NODE: case SET_COORD_5_NODE:
          case SET_COORD_6_NODE: case SET_COORD_7_NODE: case SET_COORD_8_NODE:
             { REAL v = *(stacktop--);
               REAL oldvalue = get_coord(id)[node->op2.attr_kind-SET_COORD_1_NODE];
               switch ( assign_type )
                 { case ASSIGN_OP:  break;
                   case PLUSASSIGN_OP: v +=  oldvalue; break;
                   case SUBASSIGN_OP: v = oldvalue - v; break;
                   case MULTASSIGN_OP: v = oldvalue * v; break;
                   case DIVASSIGN_OP: v = oldvalue / v; break;
                 }
               get_coord(id)[node->op2.attr_kind-SET_COORD_1_NODE] = v;
               if ( node->op2.attr_kind-SET_COORD_1_NODE <= SDIM ) *recalc_flag = 1;
             }
             break;
   
          case SET_PARAM_NODE: case SET_PARAM_1_NODE: case SET_PARAM_2_NODE:
          case SET_PARAM_3_NODE: case SET_PARAM_4_NODE:
             if ( get_vattr(id) & BOUNDARY )
               { struct boundary *boundary = get_boundary(id);
                  REAL *param = get_param(id);
                  REAL *xx = get_coord(id);
                  REAL v;
                  int pnum = node->op2.attr_kind-SET_PARAM_1_NODE; 
                  REAL oldvalue = param[pnum];
                  if ( node->right ) pnum += (int)*(stacktop--);
                  if ( pnum > boundary->pcount )
                  { sprintf(errmsg,"Parameter number is %d; maximum is %d.\n",
                          pnum,boundary->pcount);
                    sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                         file_names[node->file_no],node->line_no);
                    kb_error(1213,errmsg,RECOVERABLE);
                  }
                  v = *(stacktop--);
                  switch ( assign_type )
                  { case ASSIGN_OP:  break;
                    case PLUSASSIGN_OP: v +=  oldvalue; break;
                    case SUBASSIGN_OP: v = oldvalue - v; break;
                    case MULTASSIGN_OP: v = oldvalue * v; break;
                    case DIVASSIGN_OP: v = oldvalue / v; break;
                  }
                  param[pnum] = v; 
                  PUSH_TRACE;
                  for ( k = 0 ; k < SDIM ; k++ )
                     xx[k] = eval(boundary->coordf[k],param,id,NULL);
                  POP_TRACE;
                  *recalc_flag = 1;
               }
              else /* just parking a value probably */ 
                 { REAL *param = get_param(id);
                   int pnum = node->op2.attr_kind-SET_PARAM_1_NODE;
                   REAL oldvalue = param[pnum];
                   REAL v;
                   if ( node->right ) pnum += (int)*(stacktop--);
                  v = *(stacktop--);
                  switch ( assign_type )
                  { case ASSIGN_OP:  break;
                    case PLUSASSIGN_OP: v +=  oldvalue; break;
                    case SUBASSIGN_OP: v = oldvalue - v; break;
                    case MULTASSIGN_OP: v = oldvalue * v; break;
                    case DIVASSIGN_OP: v = oldvalue / v; break;
                  }
   
                  param[pnum] = v;
                }
                break;
   
             case SET_FRONTBODY_NODE:
             case SET_BACKBODY_NODE:
               b_id = get_ordinal_id(BODY,(int)*(stacktop--)-1);
               if ( b_id && !valid_id(b_id) ) 
               { sprintf(errmsg,"Invalid body in SET FRONTBODY or BACKBODY.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                      file_names[node->file_no],node->line_no);
                 kb_error(2032,errmsg,RECOVERABLE);
               }
               set_body((node->op2.attr_kind==SET_FRONTBODY_NODE?id:inverse_id(id)),b_id);
               *recalc_flag = 1;
               break;
   
            default: 
               sprintf(errmsg,"Unhandled SET attribute %d\n",node->op2.attr_kind);
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
               kb_error(2033,errmsg,RECOVERABLE);
               break;
   
           } /* end assigned SET_ATTRIBUTE_A */
         }
         break;

      /******************************/
      /* initializing element loops */
      /******************************/

    case INIT_VERTEX_NODE: /* all vertices */
           *(element_id *)++stacktop = q_id; /* save old element */
           *(vertex_id*)++stacktop=web.skel[VERTEX].last; /* sentinel */
           *(element_id *)++stacktop = web.skel[VERTEX].used;
           break;

    case INIT_EDGE_VERTEX_NODE: /* edge endpoints */
           *(element_id *)++stacktop = q_id; /* save old element */
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
 //          *(element_id *)++stacktop = id; /* save old element */
           *(int *)++stacktop = 0; /* element number */
           *(element_id **)++stacktop = get_edge_vertices(id);
           break;

    case INIT_FACET_VERTEX_NODE: /* facet vertices */
          *(element_id *)++stacktop = q_id; /* save old element */
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
//           *(element_id *)++stacktop = id; /* save old element */
           if ( (web.representation == SIMPLEX) ||
                   (web.modeltype==LAGRANGE) || 
                ((web.modeltype==QUADRATIC) && (web.representation==SOAPFILM)))
           { *(element_id *)++stacktop = 0 ; /* counter */
             *(element_id *)++stacktop = id ; /* facet */
           }
           else 
           { facetedge_id fe = get_facet_fe(id);
             if ( inverted(id) && web.representation==STRING && valid_id(fe) ) 
             { facetedge_id nextfe=fe,startfe=fe;
               /* go back round to find start */
               do
               { fe = nextfe;
                 nextfe = get_prev_edge(fe);
               } while ( valid_id(nextfe) && !equal_id(nextfe,startfe) );
             }
             *(element_id *)++stacktop = fe; /* start */
             *(element_id *)++stacktop = fe; /* current */
           }
           break;

    case INIT_BODY_VERTEX_NODE: /* vertices on body */
           sprintf(errmsg,"Can't do body vertices yet.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(1214,errmsg,RECOVERABLE);

           break;

    case INIT_EDGE_NODE: /* all edges */
           *(element_id *)++stacktop = q_id; /* save old element */
           *(edge_id *)++stacktop = web.skel[EDGE].last; /* sentinel */
           *(element_id *)++stacktop = web.skel[EDGE].used;
           break;

    case INIT_VERTEX_EDGE_NODE:
           *(element_id *)++stacktop = q_id; /* save old element */
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
//           *(element_id *)++stacktop = id; /* save old element */
           if ( get_vattr(id) & Q_MIDFACET )
           { *(element_id *)++stacktop = NULLID;
             *(element_id *)++stacktop = NULLID;
           }
           else
           { *(element_id *)++stacktop = get_vertex_edge(id);
             *(element_id *)++stacktop = get_vertex_edge(id);
           }
           break;
           
    case INIT_FACET_EDGE_NODE:
           *(element_id *)++stacktop = q_id; /* save old element */
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
//           *(element_id *)++stacktop = id; /* save old element */
           if ( web.representation == STRING )
           { facetedge_id fe = get_facet_fe(id);
             if ( inverted(id) && valid_id(fe) ) 
             { facetedge_id nextfe=fe,startfe=fe;
               /* go back round to find start */
               do
               { fe = nextfe;
                 nextfe = get_prev_edge(fe);
               } while ( valid_id(nextfe) && !equal_id(nextfe,startfe) );
             }
             *(element_id *)++stacktop = fe; /* start */
             *(element_id *)++stacktop = fe;
           }
           else
           { *(int *)++stacktop = 0; /* counter */
             *(element_id *)++stacktop = get_facet_fe(id);
           }
           break;
           
    case INIT_BODY_EDGE_NODE:
           sprintf(errmsg,"Can't do body edges yet.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(1215,errmsg,RECOVERABLE);
           break;

    case INIT_FACET_NODE: /* all facets */
           *(element_id *)++stacktop = q_id; /* save old element */
           *(facet_id *)++stacktop = web.skel[FACET].last; /* sentinel */
           *(element_id *)++stacktop = web.skel[FACET].used;
           break;

    case INIT_VERTEX_FACET_NODE:
 //       *(element_id *)++stacktop = q_id; /* save old element */
         if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
         else id = q_id;
//         *(element_id *)++stacktop = id; /* save old element */
         *(element_id *)++stacktop = id;  //save the vertex
         *(element_id *)++stacktop = f_id = get_vertex_first_facet(id); 
         *(element_id *)++stacktop = f_id; 
         break;

    case INIT_FACETEDGE_EDGE_NODE:
          *(element_id *)++stacktop = q_id; /* save old element */
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
//           *(element_id *)++stacktop = id; /* save old element */
           *(element_id *)++stacktop = get_fe_edge(id); /* start */
           *(element_id *)++stacktop = get_fe_edge(id);
           break;

    case INIT_FACETEDGE_FACET_NODE:
           *(element_id *)++stacktop = q_id; /* save old element */
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
//           *(element_id *)++stacktop = id; /* save old element */
           *(element_id *)++stacktop = get_fe_facet(id); /* start */
           *(element_id *)++stacktop = get_fe_facet(id);
           break;

    case INIT_EDGE_FACET_NODE:
           *(element_id *)++stacktop = q_id; /* save old element */
          if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
//           *(element_id *)++stacktop = id; /* save old element */
           *(element_id *)++stacktop = get_edge_fe(id); /* start */
           *(element_id *)++stacktop = get_edge_fe(id);
           break;

    case INIT_EDGE_FACETEDGE_NODE:
          *(element_id *)++stacktop = q_id; /* save old element */
          if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
//           *(element_id *)++stacktop = id; /* save old element */
           *(element_id *)++stacktop = get_edge_fe(id); /* start */
           *(element_id *)++stacktop = get_edge_fe(id);
           break;

    case INIT_BODY_FACET_NODE:
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
           *(element_id *)++stacktop = q_id; /* save old element */
           f_id =  get_body_facet(id);
           *(element_id *)++stacktop = f_id;
           *(element_id *)++stacktop = f_id;
           /* kludge to make things work in the face of arbitrary deletions. */
           if ( valid_id(f_id) )
           { facet_id start_f = f_id;
             int counter = 0;
             int maxcount = 2*web.skel[FACET].count;
             do
             { f_id = get_next_body_facet(f_id);
               unset_attr(f_id,DID_BODYFRONTFACET|DID_BODYBACKFACET);
               if ( counter++ > maxcount )
               { sprintf(errmsg,"Internal error: Body %s facet list is not closed.\n",
                      ELNAME(q_id));
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
                 kb_error(4376,errmsg,RECOVERABLE);
               }
             } while ( !equal_id(f_id,start_f) );
             set_attr(f_id,(inverted(f_id)?DID_BODYBACKFACET:DID_BODYFRONTFACET));
           }
           break;

    case INIT_BODY_NODE: /* all bodies */
           *(element_id *)++stacktop = q_id; /* save old element */
           *(body_id *)++stacktop = web.skel[BODY].last; /* sentinel */
           *(element_id *)++stacktop = web.skel[BODY].used;
           break;

    case INIT_FACET_BODY_NODE: 
           if ( node->op2.localnum ) 
              id = *(element_id*)get_localp(node->op2.localnum);
           else id = q_id;
           *(element_id *)++stacktop = id; /* save old element */
           *(int *)++stacktop = 0; /* element number */
           *(element_id *)++stacktop = NULLID;
           break;

    case INIT_VERTEX_BODY_NODE:
           sprintf(errmsg,"Can't do vertex bodies yet.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(1216,errmsg,RECOVERABLE);
           break;

    case INIT_EDGE_BODY_NODE:
           sprintf(errmsg,"Can't do edge bodies yet.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(1217,errmsg,RECOVERABLE);
           break;

    case INIT_FACETEDGE_NODE: /* all edges */
           *(element_id *)++stacktop = q_id; /* save old element */
           *(edge_id *)++stacktop = web.skel[FACETEDGE].last; /* sentinel */
           *(element_id *)++stacktop = web.skel[FACETEDGE].used;
           break;

    case SINGLE_ELEMENT_INIT_NODE: 
           *(element_id *)++stacktop = q_id; /* save old element */
           *(++stacktop) = 0.0 ; /* iteration count */
           stacktop++;  /* dummy local to pop */
           break;


    /*******************/
    /* simple commands */
    /*******************/

    case NOP_NODE:
    case NULLBLOCK_NODE:
    case NULLCMD_NODE:
        break;

    case DEFINE_EXTRA_NODE:
       { int one = 1; 
         struct extra *ex = EXTRAS(node->op2.eltype) + node->op1.extranum;
         if ( ex->array_spec.dim == 0 )
         { expand_attribute(node->op2.eltype,node->op1.extranum,&one);
           ex->array_spec.datacount = 1;
           ex->flags &= ~DIMENSIONED_ATTR;
         }
         break;
       }

    case DEFINE_EXTRA_INDEX_NODE:  /* have indexes tacked onto DEFINE_EXTRA_ */
       { int newsizes[MAXARRAYDIMS];
         struct extra *ex = EXTRAS(node->op2.eltype) + node->op1.extranum;
         for ( k = ex->array_spec.dim-1 ; k >= 0 ; k-- ) 
           newsizes[k] = (int)*(stacktop--);
         expand_attribute(node->op2.eltype,node->op1.extranum,newsizes);
         break;
       }

    case HISTORY_NODE:
        if ( history_space == NULL ) { outstring("No history.\n"); break;}
        for ( k = history_count-1 ; k >= 0 ; k-- )
          { h = history_space+history_offsets[k];
            sprintf(msg,"%d) ",history_number-k); outstring(msg);
            outstring(h);
          }
          break;

    case REDIRECT_NODE:
    case REDIRECTOVER_NODE:
        { struct stat st;
           int retval;
           flush_counts(); // so don't get into redirected output
           if ( node->left ) s = *(char**)(stacktop--);
           else s = node->op1.string;
           *(FILE **)(++stacktop) = outfd;
           retval = stat(s,&st);
           outfd = NULL;
           if ( ((retval == 0) && (st.st_mode & S_IFIFO)) 
            || (node->type==REDIRECTOVER_NODE) )
              outfd = fopen(s,"w");
           else outfd = fopen(s,"a");
           if ( outfd == NULL )
           { 
#ifdef MAC_APP
             sprintf(errmsg,"Cannot open file %s. \n",s);
#else
             perror(s);
             sprintf(errmsg,"Cannot open redirection file %s.\n",s);
#endif
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1218,errmsg,RECOVERABLE);

            }
           if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
             myfree(s);
          }
           break;

    case REDIRECT_END_NODE:
           fclose(outfd);
           /* outfd = stdout; */
           outfd = *(FILE**)(stacktop--);
           break;

    case REDIRECT_ERR_NODE:
    case REDIRECTOVER_ERR_NODE:
        { struct stat st;
           int retval;
           flush_counts(); // so don't get into redirected output
           if ( node->left ) s = *(char**)(stacktop--);
           else s = node->op1.string;
           *(FILE **)(++stacktop) = erroutfd;
           retval = stat(s,&st);
           erroutfd = NULL;
           if ( ((retval == 0) && (st.st_mode & S_IFIFO)) 
            || (node->type==REDIRECTOVER_NODE) )
              erroutfd = fopen(s,"w");
           else erroutfd = fopen(s,"a");
           if ( erroutfd == NULL )
           { erroutfd = stderr;
#ifdef MAC_APP
             sprintf(errmsg,"Cannot open file %s. \n",s);
#else
             perror(s);
             sprintf(errmsg,"Cannot open redirection file %s.\n",s);
#endif
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(5218,errmsg,RECOVERABLE);

            }
           if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
             myfree(s);
          }
           break;

    case REDIRECT_ERR_END_NODE:
           fclose(erroutfd);
           erroutfd = *(FILE**)(stacktop--);
           break;

    case PIPE_NODE:
           if ( node->left ) s = *(char**)(stacktop--);
           else s = node->op1.string;
           *(FILE **)(++stacktop) = outfd;
           if ( s[0] == '+' )  /* kludge to permit appending to file */
           outfd = fopen(s+1,"a+");
#ifdef NOPIPE
           else
           outfd = fopen(s,"w");
#else
           outfd = popen(s,"w");
#endif
           if ( outfd == NULL )
              { 
                outfd = *(FILE**)(stacktop--);
#ifdef MAC_APP
                sprintf(errmsg,"Cannot open pipe %s. \n",s);
#else
                perror(s);
                sprintf(errmsg,"Cannot open pipe %s.\n",s);
#endif
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                     file_names[node->file_no],node->line_no);
                kb_error(1219,errmsg,RECOVERABLE);

              }
           if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
             myfree(s);
           broken_pipe_flag = 0; /* reset */
           break;
 
    case PIPE_END_NODE:
#ifdef NOPIPE
           fclose(outfd);
#else
           pclose(outfd);
#endif
           /* outfd = stdout; */
           outfd = *(FILE**)(stacktop--);
           break;

    case DEBUG_NODE:
          outstring(yydebug?"YACC debugging was ON.":
              "YACC debugging was OFF.\n");
          yydebug = (node->op1.toggle_state==ON_) ? 1 : 0;
          outstring(yydebug?"Now ON.\n":"Now OFF.\n");
          break;

    case ITDEBUG_NODE:
          outstring(itdebug?"Iteration debugging was ON.":
              "Iteration debugging was OFF.\n");
          itdebug = (node->op1.toggle_state==ON_) ? 1 : 0;
          outstring(itdebug?"Now ON.\n":"Now OFF.\n");
          break;

    case MEMDEBUG_NODE:
          old = memdebug;
          memdebug = (node->op1.toggle_state==ON_) ? 1 : 0;
#ifdef M_DEBUG
    mallopt(M_DEBUG,memdebug);
#endif
          outstring(memdebug?"Memory debugging ON.":
                                   "Memory debugging OFF.");
          outstring(old?" (was on)\n":" (was off)\n");
          break;

    case RECALC_NODE: recalc(); break;

    case CLOSE_SHOW_NODE: close_graphics(); break;

    case TOPINFO_NODE: 
          oldquiet = quiet_flag; quiet_flag = 0;
          top_dump(outfd);
          quiet_flag = oldquiet;   
          break;

    case BOTTOMINFO_NODE: 
          oldquiet = quiet_flag; quiet_flag = 0;
          bottom_dump(outfd);
          quiet_flag = oldquiet;   
          break;

    case LIST_QUANTITY_NODE:
          list_quantity(node->op1.quant_id);
          break;

    case LIST_METHOD_INSTANCE_NODE:
          list_method_instance(node->op1.meth_id);
          break;

    case LIST_CONSTRAINT_NODE:
          k = (int)(*(stacktop--));
          if ( k < 0 || k >= web.maxcon || !(get_constraint(k)->attr & IN_USE) )
            kb_error(3614,"Invalid constraint for list_constraint.",
              RECOVERABLE);
          list_constraint(k);
          break;

    case LIST_BOUNDARY_NODE:
          k = (int)(*(stacktop--));
          if ( k < 0 || k >= web.bdrymax || !(web.boundaries[k].attr & IN_USE) )
            kb_error(3615,"Invalid boundary for list_boundary.",
              RECOVERABLE);
          list_boundary(k);
          break;

    case LIST_PROCS_NODE:
          oldquiet = quiet_flag; quiet_flag = 0;
          list_procedures(LIST_PROTO);
          quiet_flag = oldquiet;   
          break;
    
    case LIST_ATTRIBUTES_NODE:
          oldquiet = quiet_flag; quiet_flag = 0;
          outstring("//Element attributes: \n");
          list_attributes();
          quiet_flag = oldquiet;   
          break;
    
    case PRINT_ARRAY_NODE:
          oldquiet = quiet_flag; quiet_flag = 0;
          print_array(globals(node->op1.name_id)->attr.arrayptr,NULL,PRINT_PLAIN);
          quiet_flag = oldquiet;   
          break;

    case PRINT_ARRAY_LVALUE_NODE:
        { struct array *a,kludge;
          oldquiet = quiet_flag; quiet_flag = 0;
          a = get_name_arrayptr(node->op2.name_id,localstack,localbase);
          if ( node[node->left].flags & IS_VIRTUAL_ATTR )
          { kludge = *a;
            kludge.datacount = SDIM;
            kludge.sizes[0] = SDIM;
            a = &kludge;
          }
          print_array(a,*(char**)(stacktop--),PRINT_PLAIN);
          quiet_flag = oldquiet;   
          break;
        }
        
    case PRINT_ARRAYPART_NODE:
        { struct array *part;
          struct array *a;
          char *datastart = *(char**)(stacktop--);

          a = get_name_arrayptr(node[node->left].op2.name_id,localstack,localbase);

          part = (struct array *)temp_calloc(1,sizeof(struct array)+
                  a->dim*sizeof(int));
          part->dim =  a->dim - node->op5.indexcount;
          part->datatype = a->datatype;
          part->itemsize = a->itemsize;
          part->datacount = a->datacount;
          part->datastart = (char*)part - (char*)datastart;
          for ( i = node->op5.indexcount, j = 0  ; i < a->dim ; i++,j++ )
            part->sizes[j] = a->sizes[i];
          msg[0] = 0;
          oldquiet = quiet_flag; quiet_flag = 0;
          print_array(part,datastart,PRINT_PLAIN);
          quiet_flag = oldquiet;   
          temp_free((char*)part);
          break;
        }

       

    case PRINT_ATTR_ARRAY_NODE:
        { struct array *part;
          struct extra *ext;
          int offset = 0;
          n = node->op3.extranum;  /* attribute number */
          id = *(element_id*)get_localp((node+node->left)->op2.localnum);
         
          ext = EXTRAS(node->op2.eltype) + n;

          part = (struct array *)temp_calloc(1,sizeof(struct array)+
                  ext->array_spec.dim*sizeof(int));
          part->dim =  ext->array_spec.dim - node->op5.indexcount;
          part->datatype = ext->type;
          part->itemsize = ext->array_spec.itemsize;
          part->datacount = ext->array_spec.datacount;
          part->datastart =  (char*)get_extra(id,n) - (char*)part;
          for ( n = 0; n < node->op5.indexcount ; n++ )
          { int size = ext->array_spec.sizes[n];
            int inx = (int)*(stacktop - node->op5.indexcount + n + 1) - 1;
            if ( inx >= size )
            { sprintf(errmsg,
                "Index %d of attribute %s is %d; exceeds declared size, %d\n",
                n+1,ext->name,inx+1,size);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(2645,errmsg,RECOVERABLE);
            }
            if ( size ) part->datacount /= size;
            offset *= size;
            offset += inx;
          }
          stacktop -= n;
          for ( k = 0 ; n < ext->array_spec.dim ; n++,k++ )
          { int size = ext->array_spec.sizes[n];
            part->sizes[k] = size;
            offset *= size;
          }
          part->datastart += offset*part->itemsize;
          msg[0] = 0;
          oldquiet = quiet_flag; quiet_flag = 0;
          print_array(part,NULL,PRINT_PLAIN);
          quiet_flag = oldquiet;   
          temp_free((char*)part);
          break;
        }

    case DIRICHLET_NODE:
    case DIRICHLET_SEEK_NODE:
          scale = dirichlet(node->type==DIRICHLET_NODE?0:1);
          recalc();
#ifdef FLOAT128
          sprintf(msg,"%3d.  energy: %#*.*Qg  scale: %#g\n",gocount,
              DWIDTH,DPREC, web.total_energy,(DOUBLE)scale);
#elif defined(LONGDOUBLE)
          sprintf(msg,"%3d.  energy: %#*.*Lg  scale: %#g\n",gocount,
              DWIDTH,DPREC, web.total_energy,(DOUBLE)scale);
#else
          sprintf(msg,"%3d. %s: %#17.15g energy: %#17.15g  scale: %#g\n",gocount,
              areaname,web.total_area,web.total_energy,scale);
#endif 
          outstring(msg);
          break;

    case SOBOLEV_NODE:
    case SOBOLEV_SEEK_NODE:
          scale = sobolev(node->type==SOBOLEV_NODE?0:1);
          recalc();
#ifdef FLOAT128
          sprintf(msg,"%3d.  energy: %#*.*Qg  scale: %#g\n",gocount,
              DWIDTH,DPREC,web.total_energy,(DOUBLE)scale);
#elif defined(LONGDOUBLE)
          sprintf(msg,"%3d.  energy: %#*.*Lg  scale: %#g\n",gocount,
              DWIDTH,DPREC,web.total_energy,(DOUBLE)scale);
#else
          sprintf(msg,"%3d. %s: %#17.15g energy: %#17.15g  scale: %#g\n",gocount,
              areaname,web.total_area,web.total_energy,scale);
#endif 
          outstring(msg);
          break;

    case HESSIAN_SEEK_NODE:
       { REAL maxscale; 
          if ( node->left ) maxscale = *(stacktop--);
          else maxscale = 10.;
          scale = hessian_seek(maxscale);
          recalc();
#ifdef FLOAT128
          sprintf(msg,"%3d.  energy: %#*.*Qg  scale: %#g\n",gocount,
              DWIDTH,DPREC,web.total_energy,(DOUBLE)scale);
#elif defined(LONGDOUBLE)
          sprintf(msg,"%3d.  energy: %#*.*Lg  scale: %#g\n",gocount,
              DWIDTH,DPREC,web.total_energy,(DOUBLE)scale);
#else
          sprintf(msg,"%3d. %s: %#17.15g energy: %#17.15g  scale: %#g\n",
               gocount, areaname,web.total_area,web.total_energy,scale);
#endif 
          outstring(msg);
        }
        break;

    case HESSIAN_SADDLE_NODE:
       { REAL maxscale; 
          if ( node->left ) maxscale = *(stacktop--);
          else maxscale = 100.;
          hessian_saddle(maxscale);
       }
       break;

    case HESSIAN_MENU_NODE: 
          if ( hessian_subshell_flag )
          { kb_error(3633,"Can't do hessian_menu in a hessian subshell.\n",WARNING);
            break;
          }
           hessian_menu();
           recalc();
#ifdef FLOAT128
          sprintf(msg,"%3d.  energy: %#*.*Qg \n",1, 
               DWIDTH,DPREC,web.total_energy);
#elif defined(LONGDOUBLE)
          sprintf(msg,"%3d.  energy: %#*.*Lg \n",1, 
               DWIDTH,DPREC,web.total_energy);
#else
          sprintf(msg,"%3d. %s: %#17.15g energy: %#17.15g \n",1,
              areaname,web.total_area,web.total_energy);
#endif 
           outstring(msg);
           break;

    case HESSIAN_NODE: 
           hessian_auto();
           recalc();
#ifdef FLOAT128
          sprintf(msg,"%3d.  energy: %#*.*Qg  \n",gocount, DWIDTH,
               DPREC,web.total_energy);
#elif defined(LONGDOUBLE)
          sprintf(msg,"%3d.  energy: %#*.*Lg  \n",gocount, DWIDTH,
               DPREC,web.total_energy);
#else
          sprintf(msg,"%3d. %s: %#17.15g energy: %#17.15g\n",gocount,
              areaname,web.total_area,web.total_energy);
#endif 
           outstring(msg);
           break;

    case SHELL_NODE:  /* execute subshell */
#if defined(__WIN32__) || defined(_WIN32)
       _spawnlp(_P_WAIT,"cmd.exe","cmd.exe",NULL);
#else
#if defined(MAC_APP) || defined(MAC_CW)
       kb_error(1221,"No subshell on Mac.\n",RECOVERABLE);

#else
       if ( fork() == 0 )
       { execlp("sh","sh",NULL);
         perror("sh");
       }
       else wait(NULL);
#endif
#endif
          break;

    case SHOW_TRANS_NODE:
           view_transform(*(char**)(stacktop--));
           update_display();
           break;

    case CHECK_NODE:
          run_checks();
          break;

    case CONVERT_TO_QUANTS_NODE:
          if ( everything_quantities_flag )
            outstring("Everything already quantities.\n");
          else
          { convert_to_quantities();
            recalc();
          }
          break;

    case SHOW_VOL_NODE:
          show_volumes();
          break;

    case LOGFILE_NODE:
          start_logfile(*(char**)(stacktop--));
          outstring("Logfile ON.\n");
          break;

    case KEYLOGFILE_NODE:
          start_keylogfile(*(char**)(stacktop--));
          outstring("Keylogfile ON.\n");
          break;

    case GEOMVIEW_NODE:
          geomview_command(*(char**)(stacktop--));
          break;

    case IS_DEFINED_NODE:
         *stacktop = (identcase(*(char**)stacktop) != NEWIDENT_TOK);  
         break;

    case REPARTITION_NODE:
#ifdef MPI_EVOLVER
       mpi_repartition();
       update_display();
#else
       kb_error(5005,"'Repartition' command only implemented in MPI Evolver.\n",
          WARNING);
#endif
       break;

    case EXEC_NODE:
       command(*(char**)(stacktop--),NO_HISTORY);
       break;

    case PARALLEL_EXEC_NODE:
#ifdef MPI_EVOLVER
       mpi_parallel_exec(*(char**)(stacktop--));
#else
       command(*(char**)(stacktop--),NO_HISTORY);
#endif
       break;

    case TASK_EXEC_NODE:
       { int task = (int)(stacktop[-1]); 
#ifdef MPI_EVOLVER
         if ( task == 0 ) command(*(char**)(stacktop--),NO_HISTORY);
         else if ( task >= mpi_nprocs )
         { sprintf(errmsg,"Node number %d exceeds maximum node number, %d.\n",
              task,mpi_nprocs-1);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
           kb_error(3167,errmsg,RECOVERABLE);
         }
         else if ( task < 0 )
         { sprintf(errmsg,"Node number %d is negative!\n", task);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
           kb_error(3168,errmsg,RECOVERABLE);
         }
         else
           mpi_task_exec(task,*(char**)(stacktop--));
#else
         if ( task != 0 )
         { 
            sprintf(errmsg,"Node_exec node number must be 0 for non-MPI Evolver.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
            kb_error(3169,errmsg,RECOVERABLE);
         }
         command(*(char**)(stacktop--),NO_HISTORY);
#endif
         stacktop--; /* pop node number */
       }
       break;

    case SYSTEM_NODE:
       system(*(char**)(stacktop--));
       break;

    case CHDIR_NODE:
#ifdef MSC
       k = _chdir(*(char**)(stacktop--));
       if ( k < 0 )
       { sprintf(errmsg,"Unable to change directory. \n");
         if ( !strchr(*(char**)(stacktop+1),'\\') )
           strcat(errmsg,"Try using \\\\ or / instead of \\.\n");
         sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
         kb_error(2034,errmsg,RECOVERABLE);
       }
#else
       k = chdir(*(char**)(stacktop--));
       if ( k < 0 )
       { sprintf(errmsg,"Unable to change to directory \"%s\".\n",
             *(char**)(stacktop+1));
         sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
         kb_error(1116,errmsg,RECOVERABLE);
       }
#endif

       break;

    case READ_NODE:
       if ( read_depth <= 1 ) warning_messages_new = 0;
       exec_file(NULL,*(char**)(stacktop--));
       break;

    case VIEW_TRANSFORM_SWAP_COLORS_NODE: 
        { int k = (int)(*stacktop--);
          if ( (k < 1) || (k > transform_count) )
          { sprintf(errmsg,
               "Invalid index %d to view_transform_swap_colors.\n",k);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2518,errmsg, RECOVERABLE);
          }
          if ( !transform_colors ) *(++stacktop) = 0;
          else
           *(++stacktop) = (transform_colors[k-1] == SWAP_COLORS);
          break;
        }

    case VIEW_TRANSFORM_PARITY_NODE:
        { int k = (int)(*stacktop--);
          if ( !transform_parity ) 
           *(++stacktop) = 1;
          else
           *(++stacktop) = transform_parity[k-1];
          break;
        }

    case VIEW_TRANSFORMS_NODE:
       read_transforms((int)(*stacktop--));
       break;

    case VIEW_TRANSFORMS_NOP_NODE: 
         break;  /* just to get first two indices on stack */
    case VIEW_TRANSFORMS_ELEMENT_NODE:
         { int k = (int)(*stacktop--);
           int j = (int)(*stacktop--);
           int i = (int)(*stacktop--);
           if ( i < 1 )
           { sprintf(errmsg,
                 "View_transforms index 1 is %d, must be at least 1.\n",i);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2036,errmsg,RECOVERABLE);
           }
           if ( j < 1 )
           { sprintf(errmsg,
                 "View_transforms index 2 is %d, must be at least 1.\n",j);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2037,errmsg,RECOVERABLE);
           }
           if ( k < 1 )
           { sprintf(errmsg,
                 "View_transforms index 3 is %d, must be at least 1.\n",k);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2038,errmsg,RECOVERABLE);
           }
           if ( (view_transforms == 0)  && (i == 1))
           { *(++stacktop) =  (j==k) ? 1 : 0; break; }
           if ( i > transform_count )
           { sprintf(errmsg,
                 "View_transforms index 1 is %d, must be at most transform_count %d.\n",i,transform_count);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2039,errmsg,RECOVERABLE);
           }
           if ( j > 4 )
           { sprintf(errmsg,
                 "View_transforms index 2 is %d, must be at most 4.\n",j);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2040,errmsg,RECOVERABLE);
           }
           if ( k > 4 )
           { sprintf(errmsg,
                 "View_transforms index 3 is %d, must be at most 4.\n",k);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2041,errmsg,RECOVERABLE);
           }
           *(++stacktop) = view_transforms[i-1][j-1][k-1];
           break;
        }
    case TRANSFORM_DEPTH_NODE:
       generate_transforms((int)(*stacktop--));
       update_display();
       break;

    case TRANSFORM_EXPR_NODE:
       calc_view_transform_gens();
       transform_gen_expr(*(char**)(stacktop--));
       update_display();
       break;

    case DEFINE_METHOD_INSTANCE_NODE:
    case DEFINE_QUANTITY_NODE:
    case DEFINE_CONSTRAINT_NODE:
    case DEFINE_BOUNDARY_NODE:
       /* was parse-time action */
       break;

    case GRAVITY_NODE:
       old = web.gravflag;
       web.gravflag = (node->op1.toggle_state==ON_) ? 1 : 0;
       outstring(web.gravflag ? "Gravity ON." : "Gravity OFF.");
       outstring(old?" (was on)\n":" (was off)\n");
       recalc();
       break;

    case TOGGLEVALUE_NODE:
       *++stacktop = (REAL)get_toggle_value(node->op1.toggle_id);
       break;

    case FACET_COLORS_NODE:
       old = color_flag;
       color_flag = (node->op1.toggle_state==ON_) ? 1 : 0; 
       outstring(color_flag ? "Facet colors ON.":
                                        "Facet colors OFF.");
          outstring(old?" (was on)\n":" (was off)\n");
       break;

    case SHADING_NODE:
       old = shading_flag;
       shading_flag = (node->op1.toggle_state==ON_) ? 1 : 0; 
       outstring(shading_flag ? "Facet shading ON.":
                                        "Facet shading OFF.");
          outstring(old?" (was on)\n":" (was off)\n");
       break;

    case PREPRINTF_NODE:
    case PRESPRINTF_NODE:
           break;
    case GO_NODE:
          iterate();
          break;

       default:
          more_other_stuff(node,recalc_flag,
              update_display_flag,q_id,localstack,localbase);
          break;
  }

} /* end other_stuff() */

/******************************************************************
*
* Function: flip_toggle()
*
* Purpose: Utility function to flip a toggle. Put here so Mac 68K
*     can make a short jump to it within this file.
*/

void flip_toggle(
  int *flag,  /* toggle variable */
  int newstate,
  char *phrase /* to print */
)
{ int old;
  old = *flag;
  *flag = (newstate==ON_) ? 1 : 0;
  outstring(phrase);
  outstring(*flag ? " ON." : " OFF.");
  if ( old < 0 ) outstring(" (was unset)\n");
  else if ( old == 0 ) outstring(" (was off)\n");
  else outstring(" (was on)\n");
} // end flip_toggle()

/************************************************************************
*
* Function: more_other_stuff()
*
* Purpose: execute command nodes, overflow from eval() and other_stuff().
*
* Return: stacktop pointer
*
*/

void more_other_stuff(
  struct treenode * node,
  int *recalc_flag,
  int *update_display_flag,
  element_id q_id,
  REAL *localstack,
  struct locallist_t *localbase
)
{ 
  char response[100];
  int old; /* old state of toggle */
  int n,i;
  char *s;
  int oldquiet; /* old state of quiet_flag */

  struct thread_data *td = GET_THREAD_DATA;
  #define newstack  (td->eval_stack)
  #define stackmax  (td->eval_stack_size)
  #define stacktop  (td->stack_top)
  #define this_frame ((struct eval_frame*)(newstack + td->frame_spot))

  switch ( node->type )
  {
    case WRAP_VERTEX_NODE:
         { vertex_id v_id = get_ordinal_id(VERTEX,(int)(stacktop[-1])-1);
           int wrap = (int)(stacktop[0]);

           stacktop -= 2;
           if ( !web.symmetry_flag || !wrap ) break;
           wrap_vertex(v_id,wrap);
           *update_display_flag = 1;
         }
         break;

    case NORMAL_MOTION_NODE:
         old = normal_motion_flag;
         normal_motion_flag = (node->op1.toggle_state==ON_) ? 1 : 0; 
         outstring(normal_motion_flag ? "Normal motion ON.":
                                          "Normal motion OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         if ( normal_motion_flag )
             begin_normal_motion();
         else end_normal_motion();
         break;

    case VIEW_4D_NODE: 
         old = view_4D_flag;
         view_4D_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(view_4D_flag ? "4D graphics  ON." :
                                            "4D graphics  OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         update_display();
         break;
     
    case PINNING_NODE: 
         old = check_pinning_flag;
         check_pinning_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(check_pinning_flag ? "Constraint pinning ON." :
                                            "Constraint pinning OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;
     
    case METRIC_CONVERSION_NODE: 
         old = metric_convert_flag;
         metric_convert_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(metric_convert_flag ? "Metric conversion ON." :
                                            "Metric conversion OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;
     

    case SELF_SIMILAR_NODE: 
         old = self_similar_flag;
         self_similar_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(self_similar_flag ? "Self similarity mode ON." :
                                            "Self similarity mode OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case GV_BINARY_NODE: 
         old = gv_binary_flag;
         gv_binary_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(gv_binary_flag ? "Geomview binary mode ON." :
                                            "Geomview binary mode OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;
     
    case KUSNER_NODE: 
         old = kusner_flag;
         kusner_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         conf_edge_curv_flag = 0;
         outstring(kusner_flag ? "Edge square curvature ON." :
                                            "Edge square curvature OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         recalc();
         break;
     
    case BOUNDARY_CURVATURE_NODE: 
         old = boundary_curvature_flag;
         boundary_curvature_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(boundary_curvature_flag ? "Boundary curvature ON." :
                                            "Boundary curvature OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         recalc();
         break;
     
    case CONF_EDGE_SQCURV_NODE: 
         old = conf_edge_curv_flag;
         conf_edge_curv_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         kusner_flag = 0;
         outstring(conf_edge_curv_flag ? 
                "Conformal edge square curvature ON." :
                "Conformal edge square curvature OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         recalc();
         break;

    case POST_PROJECT_NODE: 
         old = post_project_flag;
         post_project_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(post_project_flag ? "Post-projection ON." :
                "Post-projection OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case MEAN_CURV_INT_NODE: 
         old = mean_curv_int_flag;
         mean_curv_int_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(mean_curv_int_flag ? "Mean curvature integral ON." :
                "Mean curvature integral OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case AUTORECALC_NODE: 
         old = autorecalc_flag;
         autorecalc_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(autorecalc_flag ? "Autorecalc ON." : "Autorecalc OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case FORCE_POS_DEF_NODE: 
         old = make_pos_def_flag;
         make_pos_def_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(make_pos_def_flag ? "Force positive definite ON." : 
                "Force positive definite OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case ESTIMATE_NODE: 
         old = estimate_flag;
         estimate_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(estimate_flag ? "Estimation ON." : "Estimation OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case TRANSFORMS_NODE: 
         old = transforms_flag;
         transforms_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(transforms_flag ? "Transform showing ON." : 
                                    "Transform showing OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         update_display();
         break;

    case DETURCK_NODE:
         old = unit_normal_flag;
         unit_normal_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(sqgauss_flag ? "Unit normal motion ON." :
                                            "Unit normal motion OFF.");
         if ( unit_normal_flag )
              { sprintf(msg,"Enter unit normal weight factor(%g): ",
                         (DOUBLE)deturck_factor);
                 prompt(msg,response,sizeof(response));
                 const_expr(response,&deturck_factor);
              }
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case SQGAUSS_NODE: 
         old = sqgauss_flag;
         sqgauss_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(sqgauss_flag ? "Squared Gaussian curvature ON." :
                                            "Squared Gaussian curvature OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         recalc();
         break;

    case STABILITY_TEST_NODE: 
         stability_test();
         break;

    case AUTOPOP_NODE: 
         old = autopop_flag;
         autopop_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(autopop_flag ? "Autopopping ON.\n" : "Autopopping OFF.\n");
         outstring(old?" (was on)\n":" (was off)\n");
         if ( autopop_flag )
            { n = 0; 
              if ( web.representation == STRING )
                 sprintf(msg,"Number of vertices popped: %d\n",
                    web.vertex_pop_count = n=verpop_str());
              else
                 sprintf(msg,"Number of vertices popped: %d\n",
                            web.vertex_pop_count = n = edgepop_film());
              outstring(msg);
              if ( n > 0 ) recalc();
            }
         break;

    case AUTOPOP_QUARTIC_NODE: 
         old = autopop_quartic_flag;
         autopop_quartic_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(autopop_quartic_flag ? "Autopop quartic mode ON.\n" :
            "Autopop quartic mode OFF.\n");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case IMMEDIATE_AUTOPOP_NODE: 
         old = immediate_autopop_flag;
         immediate_autopop_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(immediate_autopop_flag ? "Immediate autopop mode ON.\n" :
             "Immediate autopop mode OFF.\n");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case AUTOCHOP_NODE:
         old = autochop_flag;
         autochop_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         if ( autochop_flag )
         { 
           sprintf(msg,"Autochopping ON. Chop length %g ",(DOUBLE)autochop_length);
           outstring(msg);
           outstring(old?" (was on)\n":" (was off)\n");
           outstring("Set autochop length with  AUTOCHOP_LENGTH := value\n");
         }
         else 
         { outstring("Autochopping OFF.");
           outstring(old?" (was on)\n":" (was off)\n");
         }
         break;

    case UTEST_NODE: 
         simplex_delaunay_test();
         break;

    case OLD_AREA_NODE: 
         old = old_area_flag;
         old_area_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(old_area_flag ? "old_area ON." : "old_area OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case APPROX_CURV_NODE: 
         old = approx_curve_flag;
         approx_curve_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(approx_curve_flag ? "approx_curvature ON." :
                                            "approx_curvature OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case H_INVERSE_METRIC_NODE: 
         old = web.h_inverse_metric_flag;
         web.h_inverse_metric_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(web.h_inverse_metric_flag ? "h_inverse_metric ON." :
                                            "h_inverse_metric OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case ASSUME_ORIENTED_NODE: 
         old = assume_oriented_flag;
         assume_oriented_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(assume_oriented_flag ? "assume_oriented ON." :
                                            "assume_oriented OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case OOGLFILE_NODE:
         strncpy(pix_file_name,*(char**)(stacktop--),sizeof(pix_file_name));
         do_gfile('2',pix_file_name);
         break;

    case BINARY_OFF_FILE_NODE:
         strncpy(pix_file_name,*(char**)(stacktop--),sizeof(pix_file_name));
         do_gfile('7',pix_file_name);
         break;

    case POSTSCRIPT_NODE:
         if ( ps_colorflag < 0 )  ps_colorflag = 0 ;
         if ( gridflag < 0 ) gridflag = 0;
         if ( crossingflag < 0 ) crossingflag = 0;
         if ( labelflag < 0 ) labelflag = 0;
         if ( torus_display_mode == TORUS_DEFAULT_MODE ) 
         { torus_display_mode = TORUS_RAW_MODE;
           web.torus_body_flag = 0; 
           web.torus_clip_flag = 0; 
         }
         strncpy(ps_file_name,*(char**)(stacktop--),sizeof(ps_file_name));
         do_gfile('3',ps_file_name);
         break;

    case SET_CONSTRAINT_GLOBAL_NODE:
         { int connum = (int)(*stacktop--);
           struct constraint *con = get_constraint(connum);
           if ( !(con->attr & GLOBAL) )
           { vertex_id v_id;
             con->attr |= GLOBAL;
             web.con_global_map[web.con_global_count++] = (conmap_t)connum;
             FOR_ALL_VERTICES(v_id)
                set_v_constraint_map(v_id,connum);
           }
         }
         break;              

    case UNSET_CONSTRAINT_GLOBAL_NODE:
         { unsigned int connum = (int)(*stacktop--);
           struct constraint *con = get_constraint(connum);
           con->attr &= ~GLOBAL;
           for ( i = 0 ; i < web.con_global_count ; i++ )
             if ( web.con_global_map[i] == connum )
             { web.con_global_map[i] = 
                    web.con_global_map[--web.con_global_count]; 
               break;
             }
         }
         break;

    case SET_CONSTRAINT_NAME_GLOBAL_NODE:
         { int connum = node->op3.connum;
           struct constraint *con = get_constraint(connum);
           if ( !(con->attr & GLOBAL) )
           { vertex_id v_id;
             con->attr |= GLOBAL;
             web.con_global_map[web.con_global_count++] = (conmap_t)connum;
             FOR_ALL_VERTICES(v_id)
                set_v_constraint_map(v_id,connum);
           }
         }
         break;              

    case UNSET_CONSTRAINT_NAME_GLOBAL_NODE:
         { unsigned int connum = node->op3.connum;
           struct constraint *con = get_constraint(connum);
           con->attr &= ~GLOBAL;
           for ( i = 0 ; i < web.con_global_count ; i++ )
             if ( web.con_global_map[i] == connum )
             { web.con_global_map[i] = 
                    web.con_global_map[--web.con_global_count]; 
               break;
             }
         }
         break;

    case GEOMPIPE_NODE:  /* to command */
         old = geompipe_flag;
         do_gfile('C',*(char**)(stacktop--));
         break;

    case GEOMPIPE_TOGGLE_NODE: 
         old = geompipe_flag;
         do_gfile((node->op1.toggle_state==ON_) ? 'A' : 'B',NULL);
         outstring(geompipe_flag ? "geompipe ON." :
                                            "geompipe OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case GEOMVIEW_TOGGLE_NODE: 
         old = geomview_flag;
         if ( !geomview_flag && (node->op1.toggle_state==ON_) )
             do_gfile( '8',NULL);
         else if ( geomview_flag && !(node->op1.toggle_state==ON_) )
             do_gfile( '9',NULL);
         outstring(geomview_flag ? "geomview ON." :
                                            "geomview OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case LOGFILE_TOGGLE_NODE:
         old = logfile_flag;
         if ( !logfile_flag && (node->op1.toggle_state==ON_) )
             start_logfile(NULL);
         else if ( logfile_flag && (node->op1.toggle_state!=ON_) )
             stop_logfile();
         outstring(logfile_flag ? "Logfile ON." :
                                            "Logfile OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case KEYLOGFILE_TOGGLE_NODE:
         old = keylogfile_flag;
         if ( !keylogfile_flag && (node->op1.toggle_state==ON_) )
             start_keylogfile(NULL);
         else if ( keylogfile_flag && (node->op1.toggle_state!=ON_) )
             stop_keylogfile();
         outstring(keylogfile_flag ? "Keylogfile ON." :
                                            "Keylogfile OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case JIGGLE_TOGGLE_NODE: 
         old = web.jiggle_flag;
         web.jiggle_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(web.jiggle_flag ? "jiggling ON." :
                                            "jiggling OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case RIBIERE_CG_NODE: 
         old = ribiere_flag;
         ribiere_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(ribiere_flag ? "Polak-Ribiere conjugate gradient ON." :
                                            "Polak-Ribiere conjugate gradient OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         reset_conj_grad();
         break;

    case CONJ_GRAD_NODE:
         old = conj_grad_flag;
         conj_grad_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         reset_conj_grad();
         if ( conj_grad_flag )
         { 
           outstring("Conjugate gradient now ON.");
           outstring(old?" (was on)\n":" (was off)\n");
           if ( web.motion_flag )
           { sprintf(errmsg,
              "Fixed scale is ON! Not a good idea with conjugate gradient.\n");
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1223,errmsg,WARNING);
           }
         }
         else
         { 
           outstring("Conjugate gradient now OFF.");
           outstring(old?" (was on)\n":" (was off)\n");
         }
         break;

    case MEAN_CURV_NODE:
         web.norm_check_flag = 0; /* default OFF */
         old = web.area_norm_flag;
         web.area_norm_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         if ( web.area_norm_flag ) 
            { outstring("Area normalization ON.");
            }
         else
            { outstring("Area normalization OFF.");
            }
         outstring(old?" (was on)\n":" (was off)\n");
         calc_energy();  /* to make sure vertex areas set */
         break;


    case SHOW_ALL_QUANTITIES_NODE: 
          flip_toggle(&show_all_quantities,node->op1.toggle_state,"show_all_quantities");
          break;
    case PSCOLORFLAG_NODE: 
          flip_toggle(&ps_colorflag,node->op1.toggle_state,"ps_colorflag");
          break;
    case PS_CMYKFLAG_NODE: 
          flip_toggle(&ps_cmykflag,node->op1.toggle_state,"ps_cmykflag");
          break;
    case FORCE_EDGESWAP_NODE:
           flip_toggle(&force_edgeswap_flag,node->op1.toggle_state,"force_edgeswap");
          break;
    case GRIDFLAG_NODE:
          flip_toggle(&gridflag,node->op1.toggle_state,"ps_gridflag");
          break;
    case CROSSINGFLAG_NODE: 
          flip_toggle(&crossingflag,node->op1.toggle_state,"crossingflag");
          break;
    case LABELFLAG_NODE:
          flip_toggle(&labelflag,node->op1.toggle_state,"ps_labelflag");
          break;
    case BOX_FLAG_NODE:
          flip_toggle(&box_flag,node->op1.toggle_state,"show_bounding_box");
          update_display();
          break;
   case SHOW_ALL_EDGES_NODE:
          flip_toggle(&edgeshow_flag,node->op1.toggle_state,"show_all_edges");
          update_display();
          break;
    case SEPTUM_FLAG_NODE:
          flip_toggle(&septum_flag,node->op1.toggle_state,"septum_flag");
          break;
    case TORUS_FILLED_NODE:
          if ( !web.torus_flag )
          { 
            sprintf(errmsg,
             "torus_filled is invalid toggle because not in torus model.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2042,errmsg, WARNING);
            break;
          }
          flip_toggle(&web.full_flag,node->op1.toggle_state,"torus_filled");
          break;

    case VERBOSE_NODE:
         old = verbose_flag;
         verbose_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(verbose_flag ? "Verbose ON." : "Verbose OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case QUIET_NODE:
         old = quiet_flag;
         quiet_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(quiet_flag ? "Quiet ON." : "Quiet OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case QUIETLOAD_NODE:
         old = quiet_load_flag;
         quiet_load_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(quiet_load_flag ? "QuietLoad ON." : "QuietLoad OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case FUNCTION_QUANTITY_SPARSE_NODE:
         old = quantity_function_sparse_flag;
         quantity_function_sparse_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(quantity_function_sparse_flag ?
            "function_quantity_sparse ON." : "function_quantity_sparse OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case DETORUS_STICKY_NODE:
         old = detorus_sticky;
         detorus_sticky = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(detorus_sticky ?
            "detorus_sticky ON." : "detorus_sticky OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case VIEW_TRANSFORMS_USE_UNIQUE_NODE:
         old = view_transforms_unique_point_flag;
         view_transforms_unique_point_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(view_transforms_unique_point_flag ?
            "view_transforms_use_unique ON." : "view_transforms_use_unique OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case FORCE_DELETION_NODE:
         old = force_deletion;
         force_deletion = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(force_deletion ?
            "force_deletion ON." : "force_deletion OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case STAR_FINAGLING_NODE:
         old = star_finagling;
         star_finagling = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(star_finagling ?
            "star_finagling ON." : "star_finagling OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case SLICE_VIEW_NODE:
         old = slice_view_flag;
         slice_coeff_set_flag = 1;
         slice_view_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(slice_view_flag ? "slice_view ON." : "slice_view OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         update_display();
         break;

    case CLIP_VIEW_NODE:
         old = clip_view_flag;
         clip_coeff_set_flag = 1;
         clip_view_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(clip_view_flag ? "clip_view ON." : "clip_view OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         update_display();
         break;

    case BACKCULL_NODE:
         old = backcull_flag;
         backcull_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(backcull_flag ? "backcull ON." : "backcull OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         update_display();
         break;

    case ROTATE_LIGHTS_NODE:
         old = rotate_lights_flag;
         rotate_lights_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(rotate_lights_flag ? "rotate_lights ON." : "rotate_lights OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         update_display();
         break;

    case VOLGRADS_EVERY_NODE:
         old = volgrads_every_flag;
         volgrads_every_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(volgrads_every_flag ? "Volgrads_every ON." : "Volgrads_every OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case ZENER_DRAG_NODE:
         old = zener_drag_flag;
         zener_drag_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(zener_drag_flag ? "Zener_drag ON." : "Zener_drag OFF.");
         outstring(old?" (was on)":" (was off)");
         { int zd = lookup_global(ZENER_COEFF_NAME);
            if ( zd >= 0 ) 
              sprintf (msg,"  (%s: %g)\n",ZENER_COEFF_NAME,(double)globals(zd)->value.real);
            else sprintf(msg,"  (%s not set)\n",ZENER_COEFF_NAME);
            outstring(msg);
         }
         break;

    case QUIETGO_NODE:
         old = quiet_go_flag;
         quiet_go_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(quiet_go_flag ? "QuietGo ON." : "QuietGo OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case DIRICHLET_MODE_NODE:
         if ( !everything_quantities_flag )
          { convert_to_quantities();
            recalc();
          }
         flip_toggle(&dirichlet_flag,node->op1.toggle_state,"Dirichlet area mode");
         sobolev_flag = 0;
         break;

    case RGB_COLORS_FLAG_NODE:
         flip_toggle(&rgb_colors_flag,node->op1.toggle_state,"RGB colors");
         update_display();
         break;

    case BREAK_AFTER_WARNING_NODE:
         flip_toggle(&break_after_warning,node->op1.toggle_state,
          "Break after warning");
         break;

    case BREAK_ON_WARNING_NODE:
         flip_toggle(&break_on_warning,node->op1.toggle_state,
          "Break on warning");
         break;

    case BLAS_FLAG_NODE:
#ifdef BLAS
         flip_toggle(&blas_flag,node->op1.toggle_state,"using BLAS");
#else
         sprintf(errmsg,"This Evolver not compiled with BLAS.\n");
         sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
         kb_error(2472,errmsg,WARNING);
#endif
         break;

    case AUGMENTED_HESSIAN_NODE:
         flip_toggle(&augmented_hessian_flag,node->op1.toggle_state,
             "augmented Hessian");
         if ( augmented_hessian_flag )
         { if ( !sparse_constraints_flag )
             flip_toggle(&sparse_constraints_flag,node->op1.toggle_state,
                "sparse constraints");
         }
         break;

    case SPARSE_CONSTRAINTS_NODE:
         flip_toggle(&sparse_constraints_flag,node->op1.toggle_state,
                "sparse constraints");
         if ( !sparse_constraints_flag && augmented_hessian_flag )
           flip_toggle(&augmented_hessian_flag,node->op1.toggle_state,
             "augmented Hessian");
         break;

    case VISIBILITY_TEST_NODE:
         flip_toggle(&visibility_test,node->op1.toggle_state,"visibility test");
         update_display();
         break;

    case CIRCULAR_ARC_DRAW_NODE:
         flip_toggle(&circular_arc_flag,node->op1.toggle_state,"Circular arc drawing");
         break;

    case KRAYNIKPOPVERTEX_FLAG_NODE:
         flip_toggle(&kraynikpopvertex_flag,node->op1.toggle_state,"Kraynik pop vertex mode");
         break;

    case KRAYNIKPOPEDGE_FLAG_NODE:
         flip_toggle(&kraynikpopedge_flag,node->op1.toggle_state,"Kraynik pop edge mode");
         break;

    case K_ALTITUDE_FLAG_NODE:
         flip_toggle(&K_altitude_flag,node->op1.toggle_state,"K altitude mode");
         break;

    case SMOOTH_GRAPH_NODE:
         flip_toggle(&smooth_graph_flag,node->op1.toggle_state,"smooth graph");
         update_display();
         break;

    case FULL_BOUNDING_BOX_NODE:
         flip_toggle(&full_bounding_box_flag,node->op1.toggle_state,
           "full_bounding_box");
         update_display();
         break;

    case POP_TO_EDGE_NODE:
         flip_toggle(&pop_to_edge_flag,node->op1.toggle_state,"pop_to_edge");
         if ( pop_to_edge_flag ) pop_to_face_flag = 0;
         break;

    case POP_TO_FACE_NODE:
         flip_toggle(&pop_to_face_flag,node->op1.toggle_state,"pop_to_face");
         if ( pop_to_face_flag ) pop_to_edge_flag = 0;
         break;

    case POP_DISJOIN_NODE:
         flip_toggle(&pop_disjoin_flag,node->op1.toggle_state,"pop_disjoin");
         if ( pop_disjoin_flag )
           pop_enjoin_flag = 0;
         break;

    case POP_ENJOIN_NODE:
         flip_toggle(&pop_enjoin_flag,node->op1.toggle_state,"pop_enjoin");
         if ( pop_enjoin_flag )
           pop_disjoin_flag = 0;
         break;

    case BIG_ENDIAN_NODE:
         flip_toggle(&big_endian_flag,node->op1.toggle_state,"big_endian");
         if ( big_endian_flag )
           little_endian_flag = 0;
         break;

    case LITTLE_ENDIAN_NODE:
         flip_toggle(&little_endian_flag,node->op1.toggle_state,
             "little_endian");
         if ( little_endian_flag )
           big_endian_flag = 0;
         break;

    case MPI_DEBUG_NODE:
         flip_toggle(&mpi_debug,node->op1.toggle_state,"mpi_debug");
         update_display();
         #ifdef MPI_EVOLVER
         mpi_synch_mpi_debug();
         #endif
         break;

    case MPI_LOCAL_BODIES_NODE:
         flip_toggle(&mpi_debug,node->op1.toggle_state,"mpi_local_bodies");
         break;


    case BEZIER_BASIS_NODE:
       { int dim,k;
         old = bezier_flag;
         bezier_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(bezier_flag ? "bezier_basis  ON." :
             "bezier_basis OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         /* wipe old polynomials and reinitialize */
         for ( dim = 1 ; dim < MAXCOORD ; dim++ )
           for ( k = 1 ; k <= maxgaussorder[dim] ; k++ )
           { struct gauss_lag *gl = &gauss_lagrange[dim][k];
             if ( gl == NULL ) continue;
             gauss_lagrange_setup(dim,web.lagrange_order,k); 
           }
         if ( web.modeltype == LAGRANGE )
         { if ( !old && bezier_flag )
             lagrange_to_bezier();
           if ( old && !bezier_flag )
             bezier_to_lagrange();
         }
         recalc();
         break;
       }

    case SOBOLEV_MODE_NODE:
         if ( !everything_quantities_flag )
          { convert_to_quantities();
            recalc();
          }
         flip_toggle(&dirichlet_flag,node->op1.toggle_state,"Sobolev area mode");
         dirichlet_flag = 0;
         break;


    case HESSIAN_NORMAL_NODE:
         old = hessian_normal_flag;
         hessian_normal_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(hessian_normal_flag ? "hessian_normal  ON." :
             "hessian_normal OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case HESSIAN_SPECIAL_NORMAL_NODE:
         old = hessian_special_normal_flag;
         if ( node->op1.toggle_state == ON_ && 
                 hessian_special_normal_expr[0].start == NULL )
         { sprintf(errmsg,"hessian_special_normal_vector not set.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
            kb_error(3835,errmsg, RECOVERABLE);
         }
         hessian_special_normal_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(hessian_special_normal_flag ? "hessian_special_normal ON." :
             "hessian_special_normal OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case HESSIAN_NORMAL_PERP_NODE:
         old = hessian_normal_perp_flag;
         hessian_normal_perp_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(hessian_normal_perp_flag ? "hessian_normal_perp  ON." :
             "hessian_normal_perp OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;


    case HESSIAN_DOUBLE_NORMAL_NODE:
         old = hessian_double_normal_flag;
         hessian_double_normal_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(hessian_double_normal_flag ? "hessian_double_normal  ON." :
             "hessian_double_normal OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case HESSIAN_NORMAL_ONE_NODE:
         if ( node->op1.toggle_state==ON_ && 
         ((web.representation == SIMPLEX) || (SDIM - web.dimension > 1)))
         {  sprintf(errmsg,
            "HESSIAN_NORMAL_ONE only for STRING or SOAPFILM hypersurfaces.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2043,errmsg,RECOVERABLE);
         }
         old = hessian_normal_one_flag;
         hessian_normal_one_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(hessian_normal_one_flag ? "hessian_normal_one  ON." :
             "hessian_normal_one OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case HESSIAN_QUIET_NODE:
         old = hessian_quiet_flag;
         hessian_quiet_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(hessian_quiet_flag ? "hessian_quiet  ON." :
             "hessian_quiet OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case HESSIAN_DIFF_NODE:
         old = hessian_by_diff_flag;
         if ( !web.pressure_flag && count_fixed_vol() )
         {  sprintf(errmsg,
               "Hessian_diff not implemented for constrained quants.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(3902,errmsg,RECOVERABLE);
         }
         hessian_by_diff_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(hessian_by_diff_flag ? "Hessian by differences ON." :
                                            "Hessian by differences OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case INTERP_BDRY_PARAM_NODE:
         old = interp_bdry_param;
         interp_bdry_param = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(interp_bdry_param ? "interpolation of boundary parameters  ON." :
             "interpolation of boundary parameters OFF (using extrapolation).");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case RITZ_NODE:
      { int krydim;  /* Krylov subspace dimension */
         krydim = (int)*(stacktop--);
         ritz_command(*(stacktop--),krydim);
      }
      break;

    
    case LANCZOS_NODE:
      { int krydim;  /* Krylov subspace dimension */
         if ( node->right ) krydim = (int)*(stacktop--);
         else krydim = 100;
         lanczos_command(*(stacktop--),krydim);
      }
      break;

    case MOVE_NODE:
         calc_all_grads(CALC_VOLGRADS);
         move_vertices(ACTUAL_MOVE,*(stacktop--));
         vgrad_end();
#ifdef FLOAT128
         sprintf(msg,"1.  energy: %*.*Qg  stepsize: %g\n",DWIDTH,DPREC,
            web.total_energy,(DOUBLE)stacktop[1]);
#elif defined(LONGDOUBLE)
         sprintf(msg,"1.  energy: %*.*Lg  stepsize: %g\n",DWIDTH,DPREC,
            web.total_energy,(DOUBLE)stacktop[1]);
#else
         sprintf(msg,"1. %s: %17.15g energy: %17.15g  stepsize: %g\n",
            areaname,web.total_area,web.total_energy,stacktop[1]);
#endif 
         outstring(msg);
         update_display();
         break;

    case EIGENPROBE_NODE:
      { int iters;
         if ( node->right ) iters = (int)*(stacktop--);
         else iters = 0;
         eigenprobe_command(*(stacktop--),iters);
      }
      break;

    case SET_NO_DUMP_NODE:
      { if ( node->op2.intval )
          globals(node->op1.name_id)->flags2 |= NO_DUMP_BIT;
        else
          globals(node->op1.name_id)->flags2 &= ~NO_DUMP_BIT;
      }
      break;

    case EXPRLIST_NODE:
             break;  /* leave expression on stack */
 
    case GET_TRANSFORM_EXPR_NODE: /* put ptr in both halves for 32-bit */
       { int pp = (sizeof(REAL))/sizeof(char*);
         int nn;
         ++stacktop;
         for ( nn = 0 ; nn < pp  ; nn++ ) 
                ((char **)stacktop)[nn] = transform_expr;
       }
       break;

    case DATAFILENAME_NODE: /* put ptr in both halves for 32-bit */
       { int pp = (sizeof(REAL))/sizeof(char*);
         int nn;
         ++stacktop;
         for ( nn = 0 ; nn < pp  ; nn++ ) 
               ((char **)stacktop)[nn] = datafilename;
       }
       break;

    case WARNING_MESSAGES_NODE: /* put ptr in both halves for 32-bit */
       { int pp = (sizeof(REAL))/sizeof(char*);
         int nn;
         ++stacktop;
         for ( nn = 0 ; nn < pp  ; nn++ ) 
               ((char **)stacktop)[nn] = warning_messages;
       }
       break;


    case QUOTATION_NODE: /* put ptr in both halves for 32-bit */
       { int pp = (sizeof(REAL))/sizeof(char*);
         int nn;
         *++stacktop = 0.0;
         for ( nn = 0 ; nn < pp  ; nn++ ) 
                 ((char **)stacktop)[nn] = node->op1.string;
       }
       break;

    case DATE_AND_TIME_NODE:
       { time_t ltime;
         time(&ltime);
         *(char**)(++stacktop) = ctime(&ltime); 
         if ( strchr(*(char**)stacktop,'\n') ) 
            *strchr(*(char**)stacktop,'\n') = 0;
         break;
       }

    case EVOLVER_VERSION_NODE:
         *(char**)(++stacktop) = evolver_version; 
         break;
      

    case PRINTFHEAD_NODE:
    case BINARY_PRINTFHEAD_NODE:
          if ( node->op1.string ) s = node->op1.string;
          else s = *(char**)(stacktop--);
          oldquiet = quiet_flag; quiet_flag = 0;
          sprintf(msg,s);
          outstring(msg);
          quiet_flag = oldquiet;
          break;

    case ERRPRINTFHEAD_NODE:
          if ( node->op1.string ) s = node->op1.string;
          else s = *(char**)(stacktop--);
          sprintf(errmsg,s);
          erroutstring(errmsg);
          break;

    case SPRINTFHEAD_NODE:
          if ( node->op1.string )
             *(char **)(++stacktop) = node->op1.string;
          /* else already on stack */
          break;

    case PRINTF_NODE:
    case ERRPRINTF_NODE:
    case SPRINTF_NODE:
     { char format[1000];
       char *newmsg;
       int newmsgsize;
       int formatcount = 0;

       newmsgsize = 1000;
       newmsg = temp_calloc(1,newmsgsize);

       n = node[node->right].op1.argcount;
       if ( node[node->left].op1.string ) 
           s = node[node->left].op1.string;
       else
           s = *(char**)(stacktop-n);
       /* remove old string munges */
       { char *p,*c;
          /* strip %0.0s from format string */
          while ( (c = strstr(s,"%0.0s")) != NULL )
          {  for ( p = c+5 ; *p ; c++,p++ ) *c = *p;
              *c = 0;
          }
          while ( (c = strstr(s,"%.0s")) != NULL )
          {  for ( p = c+4 ; *p ; c++,p++ ) *c = *p;
              *c = 0;
          }
       }

       /* new way, parse through format string */

       {  char *msgspot = newmsg; 
          char *sp = s;
          
          while ( *sp )
          { char *f = format; /* for one format */
            int nn;
            struct treenode *nnode;
            char *ss;
            
            if ( (msgspot-newmsg) > (newmsgsize-500) )
            { size_t len = msgspot-newmsg;
              newmsg = temp_realloc(newmsg,2*newmsgsize);
              newmsgsize *= 2;
              msgspot = newmsg + len;
            }
            if ( *sp != '%' ) 
            { *(msgspot++) = *(sp++); 
              continue; 
            }
 
            /* now have % */

            /* check for %%, which reduces to % */
            if ( sp[1] == '%' )
            { *(msgspot++) = *(sp++); 
              (sp++); 
              continue; 
            }

            /* find format character */
            while ( *sp && !isalpha(*sp) ) 
              *(f++) = *(sp++);

            *(f++) = *(sp++); /* copy format character */
            *f = 0; /* null terminator */
             switch ( f[-1] )
             { case 's':
                 /* check that really have a string argument */
                 nnode = node;
                 for ( nn = 0 ; nn <= formatcount ; nn++ )
                   nnode += nnode->right;
                 nnode += nnode->left;
                 if ( !(nnode->flags &  HAS_STRING) && 
                       !(nnode->datatype == STRING_TYPE) &&
                        !(nnode->type == STRINGGLOBAL_NODE) &&
                          !(nnode->type == PERM_STRINGGLOBAL_NODE) &&
                           !(nnode->type == SPRINTFHEAD_NODE)  &&
                           !(nnode->type == DATE_AND_TIME_NODE)  &&
                           !(nnode->type == EVOLVER_VERSION_NODE) &&
                           !(nnode->type == DATAFILENAME_NODE))  
                  {  sprintf(errmsg,
                     "Argument %d: String format does not have string argument.\n",
                        formatcount+1);
                     strcat(errmsg,"Possibly % not followed by legal format?\n");
                     sprintf(errmsg+strlen(errmsg),"(source file %s, line %d; nnode->type %d)\n",
                        file_names[node->file_no],node->line_no,nnode->type);
                     kb_error(3111,errmsg,RECOVERABLE);
                  }
                 ss = *(char**)(stacktop-n+1+formatcount);
                 if ( ss == NULL ) ss = "(NULL)";
                 while ( (int)(msgspot-newmsg+strlen(ss))
                          > (int)(newmsgsize-10) )
                 { size_t len = msgspot-newmsg;
                   newmsg = temp_realloc(newmsg,2*newmsgsize);
                   newmsgsize *= 2;
                   msgspot = newmsg + len;
                 } 
                 sprintf(msgspot,format,ss);
                 ++formatcount;
                 break;
               case 'd': case 'u': case 'o': case 'x': case 'X': case 'p': case 'c':
                 sprintf(msgspot,format,(int)(*(stacktop-n+ ++formatcount)));
                 break;
               case 'f': case 'g': case 'e': case 'E': case 'G':
                 if ( fabs(*(stacktop-n+ (formatcount+1))) > 1e100 )
                    f[-1] = 'g'; /* prevent buffer overflow */
#ifdef FLOAT128
                 f[0] = f[-1]; f[-1] = 'Q'; f[1] = 0;
#elif defined(LONGDOUBLE)
                 f[0] = f[-1]; f[-1] = 'L'; f[1] = 0;
#endif
                 ++formatcount;
                 if ( !is_finite(*(stacktop-n + formatcount)) )
                    strcat(msgspot,"(NaN)");  
                 else
                    sprintf(msgspot,format,(*(stacktop-n + formatcount)));
                 break;

               case 'n': 
                 kb_error(5434,"Illegal format specifier: %n\n",RECOVERABLE);
                 break;

               default: 
                 sprintf(msgspot,format,0,0,0,0);  /* unrecognized */
             }
             msgspot += strlen(msgspot);
          }
          if ( formatcount > n )
          { sprintf(errmsg,
            "Format string has %d formats, but there are only %d arguments.\n",
              formatcount,n);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2044,errmsg,RECOVERABLE);
          }
         *msgspot = 0;   /* null terminator */
         stacktop -= n; /* pop args */
       }

       if ( node[node->left].op1.string == NULL ) stacktop--;
       if ( node->type == PRINTF_NODE )
       { int old_flag = quiet_flag;
         quiet_flag = 0;
         outstring(newmsg);
         quiet_flag = old_flag;
       }
       else if ( node->type == ERRPRINTF_NODE )
          erroutstring(newmsg);
       else /* SPRINTF_ */
       { char *str = mycalloc(strlen(newmsg)+2,1);
         strcpy(str,newmsg);
         *(char **)(++stacktop) = str;
       }
       temp_free(newmsg);
          
      }
     break;

    case BINARY_PRINTF_NODE:
     { char format[1000];
       int formatcount = 0;
       int byte_reverse = 0;
       int test_int = 0x0124567;

       if ( !big_endian_flag && !little_endian_flag )
       { sprintf(errmsg, "binary_printf: you must set 'big_endian' or 'little_endian' toggles.");
         sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
         kb_error(1935,errmsg,RECOVERABLE);
       }
       if ( big_endian_flag && (*(char*)&test_int)==0x67)
         byte_reverse = 1;
       if ( little_endian_flag && (*(char*)&test_int)==0x01)
         byte_reverse = 1;

#ifdef WIN32
       fflush(outfd);
       _setmode(_fileno(outfd),_O_BINARY);
#endif
       n = node[node->right].op1.argcount;
       if ( node[node->left].op1.string ) 
           s = node[node->left].op1.string;
       else
           s = *(char**)(stacktop-n);

       /* parse through format string */

       {  char *msgspot=NULL; 
          char *sp = s;
          
          while ( *sp )
          { char *f = format; /* for one format */
            int nn;
            struct treenode *nnode;
                    
            if ( *sp != '%' )
            { fwrite(sp,1,1,outfd);
              sp++; 
              continue; 
            }
 
            while ( *sp && (!isalpha(*sp) || *sp=='l') )
              *(f++) = *(sp++);
            *(f++) = *(sp++); /* copy format character */
            *f = 0; /* null terminator */
             switch ( f[-1] )
             { case 's':
                 /* check that really have a string argument */
                 nnode = node;
                 for ( nn = 0 ; nn <= formatcount ; nn++ )
                   nnode += nnode->right;
                 nnode += nnode->left;
                 if ( !(nnode->flags &  HAS_STRING) && 
                        !(nnode->type == STRINGGLOBAL_NODE) &&
                          !(nnode->type == PERM_STRINGGLOBAL_NODE) &&
                           !(nnode->type == SPRINTFHEAD_NODE)  &&
                           !(nnode->type == DATE_AND_TIME_NODE)  &&
                           !(nnode->type == DATAFILENAME_NODE))  
                  {  sprintf(errmsg,
                     "Argument %d: String format does not have string argument.\n",
                        formatcount+1);
                     strcat(errmsg,"Possibly % not followed by legal format?\n");
                     sprintf(errmsg+strlen(errmsg),"(source file %s, line %d; nnode->type %d)\n",
                        file_names[node->file_no],node->line_no,nnode->type);
                     kb_error(3212,errmsg,RECOVERABLE);
                  }
                 msgspot = *(char**)(stacktop-n+ ++formatcount);
                 fwrite(msgspot,1,strlen(msgspot),outfd);
                 break;
               case 'c': /* one byte */
                 { int c = (int)*(stacktop-n+ ++formatcount);
                   fwrite(&c,1,1,outfd);
                 }
                 break;
               case 'd': case 'u': case 'o': case 'x': case 'X': case 'p':
                 if ( f[-2] == 'l' )
                 { int m = (int)*(stacktop-n+ ++formatcount);
                   if ( byte_reverse )
                   { char buf[4];
                     for ( i = 0 ; i < 4 ; i++ )
                       buf[i] = ((char*)&m)[4-1-i];
                     fwrite(buf,4,1,outfd);
                   }
                   else 
                     fwrite(&m,4,1,outfd);
                 }
                 else /* 2-byte short */
                 { short m = (short)*(stacktop-n+ ++formatcount);
                   if ( byte_reverse )
                   { char buf[2];
                     for ( i = 0 ; i < 2 ; i++ )
                       buf[i] = ((char*)&m)[2-1-i];
                     fwrite(buf,2,1,outfd);
                   }
                   else 
                     fwrite(&m,2,1,outfd);
                 }
                 break;
               case 'f': case 'g': case 'e': case 'E': case 'G':
                 if ( f[-2] == 'l' )
                 { /* 8-byte double */
                   double x = *(stacktop-n+ ++formatcount);
                   if ( byte_reverse )
                   { char buf[8];
                     int i;
                     for ( i = 0 ; i < 8 ; i++ )
                       buf[i] = ((char*)&x)[7-i];
                     fwrite(buf,8,1,outfd);
                   }
                   else 
                     fwrite(&x,8,1,outfd);
                 }
                 else /* 4-byte float */
                 { float x = (float)*(stacktop-n+ ++formatcount);
                   if ( byte_reverse )
                   { char buf[4];
                     int i;
                     for ( i = 0 ; i < 4 ; i++ )
                       buf[i] = ((char*)&x)[4-i];
                     fwrite(buf,4,1,outfd);
                   }
                   else 
                     fwrite(&x,4,1,outfd);
                 }
                 break;


               default: 
                 sprintf(errmsg,"binary_printf format string has unrecognized format character '%c'\n",f[-1]);
                 break;  /* unrecognized */
             }
          }
          if ( formatcount > n )
          { sprintf(errmsg,
            "Format string has %d formats, but there are only %d arguments.\n",
              formatcount,n);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(4023,errmsg,RECOVERABLE);
          }
         stacktop -= n; /* pop args */
       }
       if ( node[node->left].op1.string == NULL ) stacktop--;
#ifdef WIN32
       fflush(outfd);
       _setmode(_fileno(outfd),_O_TEXT);
#endif
      }
     break;

    case PRINT_NODE: /* verb */
          if ( node[node->left].datatype == STRING_TYPE )
          { char *s = *(char**)(stacktop--);
            sprintf(msg,"%s\n",s);
          }
          else
          {
#ifdef FLOAT128
            sprintf(msg,"%*.*Qg\n",DWIDTH,DPREC,*(stacktop--));
#elif defined(LONGDOUBLE)
            sprintf(msg,"%*.*Lg\n",DWIDTH,DPREC,*(stacktop--));
#else
            sprintf(msg,"%20.15g\n",*(stacktop--));
#endif 
          }
          oldquiet = quiet_flag; quiet_flag = 0;
          outstring(msg);
          quiet_flag = oldquiet;
          break;

    case STRPRINT_NODE: /* verb */
          oldquiet = quiet_flag; quiet_flag = 0;
          s = *(char**)(stacktop--);
          outstring(s);
          outstring("\n");
          quiet_flag = oldquiet;   
          if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
            myfree(s);
          break;

    case PRINT_LETTER_NODE:
         oldquiet = quiet_flag; quiet_flag = 0;
         if ( single_redefine[node->op1.letter].start )
            outstring(print_express(&single_redefine[node->op1.name_id],'X'));
         else { sprintf(msg,"%c",node->op1.letter); outstring(msg);}
         outstring("\n\n");
         quiet_flag = oldquiet;   
         break;

    case PRINT_PROCEDURE_NODE:
         oldquiet = quiet_flag; quiet_flag = 0;
         outstring(print_express(&globals(node->op1.name_id)->value.proc,'X'));
         outstring("\n\n");
         quiet_flag = oldquiet;   
         break;

    case PRINT_PERM_PROCEDURE_NODE:
         oldquiet = quiet_flag; quiet_flag = 0;
         outstring(print_express(&perm_globals(node->op1.name_id)->value.proc,'X'));
         outstring("\n\n");
         quiet_flag = oldquiet;   
         break;

    case EXPRINT_PROCEDURE_NODE:
         oldquiet = quiet_flag; quiet_flag = 0;
         outstring(globals(node->op1.name_id)->attr.procstuff.proc_text);
         outstring("\n\n");
         quiet_flag = oldquiet;   
         break;

    case EPRINT_NODE: /* print and pass on value inside expression */
#ifdef FLOAT128
          sprintf(msg,"%*.*Qg\n",DWIDTH,DPREC,*stacktop);
#elif defined(LONGDOUBLE)
          sprintf(msg,"%*.*Lg\n",DWIDTH,DPREC,*stacktop);
#else
          sprintf(msg,"%20.15g\n",*stacktop);
#endif 
          oldquiet = quiet_flag; quiet_flag = 0;
          outstring(msg);
          quiet_flag = oldquiet;   
          break;

    case SHOWQ_NODE:
         go_display_flag = 1;
         display();
         break;

    case SET_THICKEN_NODE:
      thickness = *(stacktop--);
      break;

    case JIGGLE_NODE:
      web.temperature = *(stacktop--);
      jiggle();
      recalc();
      break;
    
    case QUIT_NODE:
      exit((int)floor(*stacktop));
      break;

    case NOTCH_NODE:
      if ( web.representation == SIMPLEX )
      { sprintf(errmsg,"Notching not implemented for simplex representation.\n");
        sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
        kb_error(1224,errmsg,RECOVERABLE);
      }
      web.max_angle = *(stacktop--);
      if ( web.max_angle <= 0.0 )
      {  sprintf(errmsg,"Notching angle not positive.\n");
         sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
         kb_error(1225,errmsg,RECOVERABLE);
      }
      if ( web.representation == STRING )
        { sprintf(msg,"refine edge ee where max(ee.vertex,dihedral) > %f",
              (DOUBLE)web.max_angle);
          command(msg,NO_HISTORY);
          web.notch_count = web.edge_refine_count;
        }
      else 
        { web.notch_count = ridge_notcher(web.max_angle);
          sprintf(msg,"Number of edges notched: %d\n",web.notch_count);
          outstring(msg);
          recalc();
        }
      break;

    case SET_AUTOCHOP_NODE:
      autochop_length = *(stacktop--);
      break;

    case SET_AMBIENT_PRESSURE_NODE:
      web.pressure = *(stacktop--);
      if ( web.pressure > 0.00000001 )
         {  body_id b_id;
            if ( !web.full_flag && !valid_id(web.outside_body) )
                              add_outside();
            web.projection_flag = 0;
            web.pressure_flag = 1;
            if ( everything_quantities_flag )
            { FOR_ALL_BODIES(b_id) 
                create_pressure_quant(b_id);
            }
         }
      else
         {
            web.projection_flag = 1;
            web.pressure_flag = 0;
         }
      recalc();
      break;

    case SET_DIFFUSION_NODE:
         old = web.diffusion_flag;
         web.diffusion_const = *(stacktop--);
         web.diffusion_flag = 1;
         outstring(web.diffusion_flag ? "Diffusion ON." :
                                            "Diffusion OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case AREAWEED_NODE:
         sprintf(msg,"Skinny triangles weeded: %d\n",
              web.facet_delete_count = areaweed(*(stacktop--)));
         outstring(msg);
         energy_init = 0;
         recalc();
         break;

    case METIS_NODE:
    case KMETIS_NODE:
/*
         metis_partition_dual((int)(*(stacktop--)),node->type);
*/
         metis_partition_plain((int)(*(stacktop--)),node->type);
         break;

    case METIS_READJUST_NODE:
         metis_partition_dual((int)(*(stacktop--)),METIS_READJUST);
/*
         metis_partition_plain((int)(*(stacktop--)),METIS_READJUST);
*/
         break;

    case BODY_METIS_NODE:
         metis_partition_body((int)(*(stacktop--)),METIS_MODE);
         break;

    case OMETIS_NODE:
         if ( node->left )
            metis_vertex_order((int)(*(stacktop--)));
         else metis_vertex_order(100);
         break;

    case EDGEWEED_NODE:
         sprintf(msg,"Deleted edges: %d\n",
              web.edge_delete_count = edgeweed(*(stacktop--)));
         outstring(msg);
         recalc();
         break;

    case OPTIMIZE_NODE:
         web.motion_flag = (node->op1.toggle_state==ON_) ? 0 : 1;
         if ( web.motion_flag )
            sprintf(msg,"Scale fixed at %g.\n",(DOUBLE)web.scale);
         else sprintf(msg,"Scale optimizing with bound %g.\n",(DOUBLE)web.maxscale);
         outstring(msg);
         break;

    case SET_OPTIMIZE_NODE:
         web.maxscale = *(stacktop--);
         web.motion_flag = 0;
         sprintf(msg,"Scale optimizing with bound %g.\n",(DOUBLE)web.maxscale);
         outstring(msg);
         break;

    case SET_SCALE_NODE:
         web.scale = *(stacktop--);
         web.motion_flag = 1;
         sprintf(msg,"Scale fixed at %g.\n",(DOUBLE)web.scale);
         outstring(msg);
         break;

    case SET_GRAVITY_NODE:
         switch ( node->op1.assigntype )
         { case ASSIGN_OP: web.grav_const = *(stacktop--); break;
           case PLUSASSIGN_OP: web.grav_const += *(stacktop--); break;
           case SUBASSIGN_OP: web.grav_const -= *(stacktop--); break;
           case MULTASSIGN_OP: web.grav_const *= *(stacktop--); break;
           case DIVASSIGN_OP: web.grav_const /= *(stacktop--); break;
         }
         old = web.gravflag;
         if ( web.grav_const != 0.0 )
         { web.gravflag = 1;
           sprintf(msg,"Gravity is now ON with gravitational constant %g.",
              (DOUBLE)web.grav_const);
         }
         else
         { web.gravflag = 0;
           sprintf(msg,"Gravity is now OFF.");
         }
         if (gravity_quantity_num >= 0 )
           GEN_QUANT(gravity_quantity_num)->modulus =
                              web.gravflag ? web.grav_const : 0.0;
         outstring(msg);
         outstring(old?" (was on)\n":" (was off)\n");
         recalc();
         break;

    case SET_MODEL_NODE:
         switch ( (int)*(stacktop--) )
         { 
            case LINEAR: 
                if ( web.modeltype == QUADRATIC )
                { outstring("Changing to LINEAR model. (was QUADRATIC)\n");
                  quad_to_linear(); 
                }
                else if ( web.modeltype == LAGRANGE )
                 { outstring("Changing to LINEAR model. (was LAGRANGE)\n");
                   lagrange_to_linear(); 
                 }
                 else outstring("Model already LINEAR.\n");
                 break;
            case QUADRATIC: 
                if ( web.modeltype == LINEAR )
                 { outstring("Changing to QUADRATIC model. (was LINEAR)\n");
                   linear_to_quad();  
                }
                else if ( web.modeltype == LAGRANGE )
                { outstring("Changing to QUADRATIC model. (was LAGRANGE)\n");
                    lagrange_to_quad(); 
                }
                else outstring("Model already QUADRATIC.\n");
                 break;
            default: 
                 if ( stacktop[1] > 2. )
                 { if ( web.modeltype == LINEAR )
                   { outstring("Changing to LAGRANGE model. (was LINEAR)\n");
                     linear_to_lagrange((int)stacktop[1]); 
                   }
                   else if ( web.modeltype == QUADRATIC )
                   { outstring("Changing to LAGRANGE model. (was QUADRATIC)\n");
                     quad_to_lagrange((int)stacktop[1]); 
                   }
                   else 
                   { sprintf(msg,
                       "Changing to LAGRANGE %d model. (was LAGRANGE %d)\n",
                           (int)stacktop[1],web.lagrange_order);
                     outstring(msg); 
                     lagrange_to_lagrange((int)stacktop[1]); 
                   }
                 }
                 else
            {  sprintf(errmsg,
        "Bad model choice. Legal: 1 (linear), 2 (quadratic), > 2 (Lagrange)\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(1230,errmsg,RECOVERABLE);
           }
         }
         recalc();
         break;

    case INVOKE_P_MENU_NODE:
         display_file((int)*(stacktop--));
         break;

    case EDGEDIVIDE_NODE:
         sprintf(msg,"New edges: %d\n",
             web.edge_refine_count = articulate(*(stacktop--)));
         outstring(msg);
         recalc();
            break;

    case SET_SGLOBAL_NODE:
       { struct global *g = globals(node->op1.name_id);
         char *s,**ss;

         s = *(char**)(stacktop--);         
         if ( g->flags & GLOB_LOCALVAR )
         { ss = (char**)(localstack+g->value.offset); 
           *ss = temp_realloc(*ss,(s?strlen(s):0)+1);
         }
         else 
         { ss = &(g->value.string);
           *ss = kb_realloc(*ss,(s?strlen(s):0)+1);
         }
         strcpy(*ss,s);
         g->flags |= STRINGVAL;
         if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
           myfree(s);  // for sprintf source of right side
       } break;

    case SET_PERM_SGLOBAL_NODE:
       { struct global *g = perm_globals(node->op1.name_id);
         char *s;
         if ( g->value.string && !(g->flags & INTERNAL_NAME) ) 
             free(g->value.string);
         s = *(char**)(stacktop--);
         if ( s )
         { if ( !(g->flags & INTERNAL_NAME) )
              g->value.string = calloc(strlen(s)+1,sizeof(char));
           strcpy(g->value.string,s);
           #ifdef WIN32
           if ( node->op1.name_id == console_title_global )
             SetConsoleTitleA(console_title);
           #endif
           if ( node->op1.name_id == graphics_title_global )
             set_graphics_title(1, graphics_title);
           else if ( node->op1.name_id == graphics_title2_global )
             set_graphics_title(2, graphics_title2);
           else if ( node->op1.name_id == graphics_title3_global )
             set_graphics_title(3, graphics_title3);         
         }
         else 
         { g->value.string = calloc(1,sizeof(char));  /* empty */
         } 
         if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
           myfree(s);

         g->flags |= STRINGVAL;
       } break;


    case SET_PARAM_SCALE_NODE:
       { struct global *g = globals(node->op1.name_id);
         REAL value = *(stacktop--);
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: g->attr.varstuff.pscale = value; break;
           case PLUSASSIGN_OP: g->attr.varstuff.pscale += value; break;
           case SUBASSIGN_OP: g->attr.varstuff.pscale -= value; break;
           case MULTASSIGN_OP: g->attr.varstuff.pscale *= value; break;
           case DIVASSIGN_OP: g->attr.varstuff.pscale /= value; break;
         } 
         break;
       }

    case SET_DELTA_NODE:
       { struct global *g = globals(node->op1.name_id);
         REAL value = *(stacktop--);
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: g->attr.varstuff.delta = value; break;
           case PLUSASSIGN_OP: g->attr.varstuff.delta += value; break;
           case SUBASSIGN_OP: g->attr.varstuff.delta -= value; break;
           case MULTASSIGN_OP: g->attr.varstuff.delta *= value; break;
           case DIVASSIGN_OP: g->attr.varstuff.delta /= value; break;
         } 
         break;
       }

    case SET_ON_ASSIGN_CALL_NODE:
       { struct global *g = globals(node->op1.name_id);
         g->attr.varstuff.on_assign_call = node->op2.name_id;
         break;
       }
 

    case DEFINE_IDENT_NODE: break;

    case DEFINE_ARRAY_NODE:
          { int dim,size,i; 
            struct treenode *nnode = node;
            struct global *g = globals(node->op1.name_id);
            int itemsize;
            int pointercount;
            struct array *oldarray;
            struct array **array_info;
            int different_size;

            if ( g->flags & GLOB_LOCALVAR )
            { localbase->flags |= LL_HAS_ARRAY;
              array_info = (struct array **)(localstack+g->value.offset);
            }
            else array_info = &(g->attr.arrayptr);
            
            oldarray = *array_info;
            itemsize = datatype_size[node->op2.valtype];
            pointercount = 0;
            different_size = oldarray ? 0 : 1;
            for ( size = 1, dim=0 ; dim < g->attr.arrayptr->dim ; dim++ ) 
            { int dimsize;
              pointercount += size;
              dimsize = (int)stacktop[1-g->attr.arrayptr->dim+dim];
              size *= dimsize;
              if ( oldarray && (dimsize != oldarray->sizes[dim]) )
                different_size = 1;
            }
            if ( oldarray && (oldarray->datatype != node->op2.valtype) )
            { different_size = 1;
              if ( oldarray->datatype )
              { sprintf(errmsg,"Changing datatype of array '%s'.  Data zeroed.\n",g->name);
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                   file_names[node->file_no],node->line_no);
                kb_error(4822,errmsg,WARNING);
              }
              oldarray = NULL;  // zero out data instead of converting
            }
            if ( !different_size )
            { stacktop -= dim;
              break;  // no need to do anything
            }
           
            if ( g->flags & GLOB_LOCALVAR )
              *array_info = (struct array*)temp_calloc( 
                sizeof(struct array)+dim*sizeof(int)
                   + (size+1)*itemsize+pointercount*sizeof(REAL*),1);  /* extra for alignment */
            else /* nonlocal */
              *array_info = (struct array*)mycalloc( 
                sizeof(struct array)+dim*sizeof(int)
                           + (size+1)*itemsize+pointercount*sizeof(REAL*),1);  /* extra for alignment */
            (*array_info)->dim = dim;
            (*array_info)->itemsize = itemsize;
            (*array_info)->datatype = node->op2.valtype;
            (*array_info)->datacount = size;
            for ( i=0, nnode = node ; i < dim ; nnode += nnode->left, i++ ) 
              (*array_info)->sizes[i] = (int)stacktop[1-dim+i];
            stacktop -= dim;
            (*array_info)->datastart = sizeof(struct array) + dim*sizeof(int);  
            /* guarantee bus alignment */
            if ( (*array_info)->datastart % itemsize )
              (*array_info)->datastart = 
                 (*array_info)->datastart + itemsize - 
                    ((*array_info)->datastart % itemsize ); 

            /* copy old data */ 
            if ( oldarray )
            { int oldoff,n;
              for ( oldoff = 0 ; oldoff < oldarray->datacount ; oldoff ++ )
              { int newoff = 0;
                int prod = 1;
                int p,pp;
                p = oldoff;
                for ( n = dim-1 ; n >= 0 ; n-- )
                { int nd = (*array_info)->sizes[n];
                  int od = oldarray->sizes[n];
                  if ( od == 0 ) goto skip;
                  pp = p % od;
                  p  = p / od;
                  if ( pp >= nd ) goto skip;
                  newoff += prod*pp;
                  prod *= nd; 
                }
                memcpy((char*)(*array_info)+(*array_info)->datastart
                          + newoff*(*array_info)->itemsize,
                        (char*)oldarray+oldarray->datastart
                           + oldoff*oldarray->itemsize,
                         oldarray->itemsize);
skip: ;
              }
            }
            
            if ( g->flags & GLOB_LOCALVAR )
              temp_free((char*)oldarray);
            else 
              myfree((char*)oldarray);
           } 
           break;
         
    case SET_ELEMENT_GLOBAL_NODE:
       { struct global *g = globals(node->op1.name_id);
         g->value.id = *(element_id*)(stacktop--);
         break;
       } 
          

    case ZOOM_NODE:
            if ( node->left ) /* have vertex number */
              { vertex_id v_id;
                 int vnum;
                 int found = 0;
                 web.zoom_radius = *(stacktop--);
                 vnum = (int)*(stacktop--);
                 FOR_ALL_VERTICES(v_id)
                 { if ( vnum == (ordinal(v_id)+1) )
                   { web.zoom_v = v_id; found = 1; break; }
                 }
                 if ( !found )
                 { sprintf(errmsg,"Zoom vertex %d not found.\n",vnum);
                   sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                     file_names[node->file_no],node->line_no);
                   kb_error(1231,errmsg,RECOVERABLE);
                 }
              }
            else if ( !valid_element(web.zoom_v) )
                 { sprintf(errmsg,"Zoom vertex not found.\n");
                   sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                     file_names[node->file_no],node->line_no);
                   kb_error(1232,errmsg,RECOVERABLE);
                 }

            zoom_vertex(web.zoom_v,web.zoom_radius);
            /* resize(); */
            recalc();
            break;

    case VIEW_MATRIX_LVALUE_NODE: break;  /* just to hold indices */

    case SET_VIEW_MATRIX_NODE:
      { int i = (int)(stacktop[-2]);
        int k = (int)(stacktop[-1]);
        REAL value = *stacktop;
        stacktop -= 3;
         if ( (k < 1) || (k > SDIM+1) || (i < 1) || (i > SDIM+1) )
         { sprintf(errmsg,
             "Illegal index: view_matrix[%d][%d] (must be 1 to %d)\n",i,k,SDIM);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2049,errmsg,RECOVERABLE);
         }
         i--;k--;  /* convert to 0 based indexing */
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: view[i][k] = value; break;
           case PLUSASSIGN_OP: view[i][k] += value; break;
           case SUBASSIGN_OP: view[i][k] -= value; break;
           case MULTASSIGN_OP: view[i][k] *= value; break;
           case DIVASSIGN_OP: 
               if ( value == 0.0 )
               { sprintf(errmsg,"Division by zero.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2565,errmsg,RECOVERABLE);
               }
               view[i][k] /= value; break;
         } 
         *update_display_flag = 1;
      }
      break;


    case SET_QTARGET_NODE:
      { struct gen_quant *q = GEN_QUANT(node->op1.quant_id);
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: q->target = *(stacktop--); break;
           case PLUSASSIGN_OP: q->target += *(stacktop--); break;
           case SUBASSIGN_OP: q->target -= *(stacktop--); break;
           case MULTASSIGN_OP: q->target *= *(stacktop--); break;
           case DIVASSIGN_OP:
               if ( *stacktop == 0.0 )
               { sprintf(errmsg,"Division by zero.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2588,errmsg,RECOVERABLE);
               }
               q->target /= *(stacktop--); 
               break;
         } 
         if ( valid_id(q->b_id) )
              set_body_fixvol(q->b_id,q->target);
      }
      break;

    case SET_QMODULUS_NODE:
      { REAL v = GEN_QUANT(node->op1.quant_id)->modulus;
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: v = *(stacktop--); break;
           case PLUSASSIGN_OP: v += *(stacktop--); break;
           case SUBASSIGN_OP: v -= *(stacktop--); break;
           case MULTASSIGN_OP: v *= *(stacktop--); break;
           case DIVASSIGN_OP: 
              if ( *stacktop == 0.0 )
               { sprintf(errmsg,"Division by zero.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2589,errmsg,RECOVERABLE);
               }
              v /= *(stacktop--); 
              break;
         } 
        GEN_QUANT(node->op1.quant_id)->modulus = v;
     }
     break;

    case SET_QTOLERANCE_NODE:
      if ( *stacktop <= 0.0 )
      { sprintf(errmsg,"Tolerance must be positive.\n");
        sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
         kb_error(2050,errmsg,RECOVERABLE);
      }
      { REAL v = GEN_QUANT(node->op1.quant_id)->tolerance;
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: v = *(stacktop--); break;
           case PLUSASSIGN_OP: v += *(stacktop--); break;
           case SUBASSIGN_OP: v -= *(stacktop--); break;
           case MULTASSIGN_OP: v *= *(stacktop--); break;
           case DIVASSIGN_OP:
             if ( *stacktop == 0.0 )
               { sprintf(errmsg,"Division by zero.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2487,errmsg,RECOVERABLE);
               }
             v /= *(stacktop--); 
             break;
         } 
        GEN_QUANT(node->op1.quant_id)->tolerance = v;
     }
     break;

    case SET_MMODULUS_NODE:
      { REAL v = METH_INSTANCE(node->op1.meth_id)->modulus;
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: v = *(stacktop--); break;
           case PLUSASSIGN_OP: v += *(stacktop--); break;
           case SUBASSIGN_OP: v -= *(stacktop--); break;
           case MULTASSIGN_OP: v *= *(stacktop--); break;
           case DIVASSIGN_OP:
             if ( *stacktop == 0.0 )
               { sprintf(errmsg,"Division by zero.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2488,errmsg,RECOVERABLE);
               }
             v /= *(stacktop--); 
             break;
         } 
        METH_INSTANCE(node->op1.meth_id)->modulus = v;
     }
     break;

    case SET_QVOLCONST_NODE:
      { struct gen_quant *q =
             GEN_QUANT(node->op1.quant_id);
         switch ( node->op2.assigntype )
         { case ASSIGN_OP: q->volconst = *(stacktop--); break;
           case PLUSASSIGN_OP: q->volconst += *(stacktop--); break;
           case SUBASSIGN_OP: q->volconst -= *(stacktop--); break;
           case MULTASSIGN_OP: q->volconst *= *(stacktop--); break;
           case DIVASSIGN_OP:
             if ( *stacktop == 0.0 )
               { sprintf(errmsg,"Division by zero.\n");
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(2489,errmsg,RECOVERABLE);
               }
             q->volconst /= *(stacktop--); 
             break;
         } 
         if ( valid_id(q->b_id) )
              set_body_volconst(q->b_id,q->volconst);
      }
    break;

    case RUNGE_KUTTA_NODE:
         old = runge_kutta_flag;
         runge_kutta_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(runge_kutta_flag ? "Runge-Kutta ON." :
                                            "Runge-Kutta OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case CHECK_INCREASE_NODE: 
         old = check_increase_flag;
         check_increase_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(check_increase_flag?"Increase check ON.":
                          "Increase check OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         break;

    case HOMOTHETY_NODE:
         old = web.homothety;
         web.homothety = (node->op1.toggle_state==ON_) ? 1 : 0;
         if ( web.homothety && (web.skel[BODY].count == 0) )
          { web.homothety = 0;
            sprintf(errmsg,"Cannot do homothety without bodies.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(1233,errmsg,RECOVERABLE);
          }
         sprintf(msg,"Homothety adjustment is %s.",
                          web.homothety ? "ON" : "OFF");
         outstring(msg);
         outstring(old?" (was on)\n":" (was off)\n");
         if ( web.homothety )
            { sprintf(msg,"Enter target size (%g): ",(DOUBLE)homothety_target);
              prompt(msg,response,sizeof(response));
              const_expr(response,&homothety_target);
            }
         break;

    case COUNTS_NODE: /* report count of elements and status */
         memory_report();
         break;

    case EXTRAPOLATE_NODE: extrapolate(); break;

    case DIFFUSION_NODE: /* Set diffusion */
         old = web.diffusion_flag;
         web.diffusion_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(web.diffusion_flag ? "Diffusion ON." :
                                            "Diffusion OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
            break;

    case AUTODISPLAY_NODE: 
      old = go_display_flag;
      go_display_flag = (node->op1.toggle_state == ON_) ? 1 : 0;
      outstring(go_display_flag ? "Autodisplay ON." :
                                            "Autodisplay OFF.");
      outstring(old?" (was on)\n":" (was off)\n");
      if ( go_display_flag ) update_display();
      break;

    case SHOW_INNER_NODE: 
      old = innerflag;
      innerflag = (node->op1.toggle_state == ON_) ? 1 : 0;
      outstring(innerflag ? "show inner ON." : "show inner OFF.");
      outstring(old?" (was on)\n":" (was off)\n");
      update_display();
      break;

    case SHOW_OUTER_NODE: 
      old = outerflag;
      outerflag = (node->op1.toggle_state == ON_) ? 1 : 0;
      outstring(outerflag ? "show outer ON." : "show outer OFF.");
      outstring(old?" (was on)\n":" (was off)\n");
      update_display();
      break;

    case INTERP_NORMALS_NODE: 
      normflag = (node->op1.toggle_state == ON_) ? 1 : 0;
      break;

    case COLORMAP_NODE: 
      colorflag = (node->op1.toggle_state == ON_) ? 1 : 0;
      break;

    case AMBIENT_PRESSURE_NODE: 
      web.pressure_flag = (node->op1.toggle_state == ON_) ? 1 : 0;
      if ( web.pressure_flag)
       { if ( !web.full_flag && !valid_id(web.outside_body) )
             add_outside();
         web.projection_flag = 0;
         sprintf(msg,"Ambient pressure ON; ambient pressure = %2.15g\n",
                    (double)web.pressure);
       }
      else
       {
         web.projection_flag = 1;
         sprintf(msg,"Ambient pressure OFF.\n");
       }
      calc_energy();
      outstring(msg);
      break;

    case EFFECTIVE_AREA_NODE:
         if ( SDIM > 3 )
         { sprintf(errmsg,"effective_area only for dimension 3.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2051,errmsg,RECOVERABLE);
         }
         old = effective_area_flag;
         effective_area_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(effective_area_flag ? "effective_area ON." :
                                            "effective_area OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         if ( effective_area_flag ) web.area_norm_flag = 1;
         if ( square_curvature_flag ) calc_energy();
         break;

    case NORMAL_CURVATURE_NODE:
         if ( SDIM != 3 )
         { sprintf(errmsg,"Normal_curvature only for dimension 3.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2052,errmsg,RECOVERABLE);
         }
         old = normal_curvature_flag;
         normal_curvature_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(normal_curvature_flag ? "normal_curvature ON." :
                                            "normal_curvature OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         if ( square_curvature_flag ) calc_energy();
         break;

    case DIV_NORMAL_CURVATURE_NODE:
         if ( SDIM != 3 )
         { sprintf(errmsg,"Div_normal_curvature only for dimension 3.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2053,errmsg,RECOVERABLE);
         }
         old = div_normal_curvature_flag;
         div_normal_curvature_flag = (node->op1.toggle_state==ON_) ? 1 : 0;
         outstring(div_normal_curvature_flag ? "div_normal_curvature ON." :
                                            "div_normal_curvature OFF.");
         outstring(old?" (was on)\n":" (was off)\n");
         if ( square_curvature_flag ) calc_energy();
         break;

    case LONG_JIGGLE_NODE: 
         long_jiggle();
         break;

    case RAW_VERAVG_NODE:
         vertex_average(NOVOLKEEP);
         recalc();
         outstring("Vertex averaging done.\n");
         break;

    case RAWEST_VERAVG_NODE:
         vertex_average(RAWEST);
         recalc();
         outstring("Vertex averaging done.\n");
         break;

    case TEXT_SPOT_NODE: /* just accumulate arguments */
    case TEXT_SIZE_NODE:
         break;

    case DISPLAY_TEXT_NODE:
         { char *text = *(char**)(stacktop--);
           REAL text_size = *(stacktop--);
           REAL yspot = *(stacktop--);
           REAL xspot = *(stacktop--);

           int length = (int)strlen(text);
           int text_id;
           
           for ( text_id = 0 ; text_id < MAXTEXTS ; text_id++ )
           { if ( text_chunks[text_id].text == NULL )
             { text_chunks[text_id].text =
                 (char*)mycalloc(length+1,1);
               break;
             }
           }
             
           if ( text_id >= MAXTEXTS-1 )
           { sprintf(errmsg,"Too many display texts.\n");
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
             kb_error(4998,errmsg,WARNING);
             break;
           }
           
           strcpy(text_chunks[text_id].text,text);
           text_chunks[text_id].start_x = xspot;
           text_chunks[text_id].start_y = yspot;
           text_chunks[text_id].vsize = text_size;
           display_text_count++;
           *(++stacktop) = text_id+1;
           *recalc_flag = 1;
           break;
         }
           
    case DELETE_TEXT_NODE:
      { int text_id = (int)*(stacktop--) - 1;
        if ( (text_id < 0) || (text_id >= MAXTEXTS) )
        { sprintf(errmsg,"Text id must be between 1 and %d.\n",MAXTEXTS);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
          kb_error(7543,errmsg,RECOVERABLE);
        }
        if ( text_chunks[text_id].text )
        { myfree(text_chunks[text_id].text);
          text_chunks[text_id].text = NULL;
          display_text_count--;
          *recalc_flag = 1;
        }
        break;  
      }
      
    case REBODY_NODE:
      { body_id b_id;
        sprintf(msg,"New bodies: %d\n",rebody());
        outstring(msg);
        sprintf(msg,"Merged bodies: %d\n",merge_bodies());
        outstring(msg);
        calc_content(Q_FIXED|Q_INFO);
        FOR_ALL_BODIES(b_id)
            if ( get_battr(b_id) & FIXEDVOL )  
              set_body_fixvol(b_id,get_body_volume(b_id));
         break;
      } 

    case SUPPRESS_WARNING_NODE:
        { int wnum = (int)*(stacktop--);
          // check for existing suppression
          for ( i = 0 ; i < warnings_suppressed_count ; i++ )
            if ( warnings_suppressed[i] == wnum )
              break;
          if ( i == warnings_suppressed_count )
          { // need to install new suppression
            if ( warnings_suppressed_count < MAXSUPPRESS )
              warnings_suppressed[warnings_suppressed_count++] = wnum;
            else
              kb_error(4535,"Too many warnings suppressed.\n",WARNING);
          }
        }
        break;

    case UNSUPPRESS_WARNING_NODE:
        { int wnum = (int)*(stacktop--);
           for ( i = 0 ; i < warnings_suppressed_count ; i++ )
             if ( warnings_suppressed[i] == wnum )
               { warnings_suppressed[i] = 
                      warnings_suppressed[--warnings_suppressed_count];
                 break;
               }          
         }
         break; 

    case ALICE_NODE:      
         alice();
         break;

    case BURCHARD_NODE:      
         burchard(node->op1.maxsteps);
         break;

    case DUMP_NODE:
         if ( node->left ) do_dump(*(char**)(stacktop--));
         else do_dump(NULL);
         break;

    case SET_COLORMAP_NODE:
         strncpy(cmapname,*(char**)(stacktop--),sizeof(cmapname));
         break;

    case RAW_CELLS_NODE: 
         web.torus_body_flag = 0; web.torus_clip_flag = 0;
         torus_display_mode = TORUS_RAW_MODE;
         update_display();
         break;

    case CONNECTED_CELLS_NODE: 
         if ( web.skel[BODY].count == 0 )
         {  sprintf(errmsg,"There are no bodies to display connectedly.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(1234,errmsg,WARNING);
         }
         else { web.torus_body_flag = 1; web.torus_clip_flag = 0; 
                  torus_display_mode = TORUS_CONNECTED_MODE;
                  update_display(); }
         break;

    case CLIPPED_CELLS_NODE:
         if ( !web.torus_flag && !web.torus_display_period ) 
         {  sprintf(errmsg,
               "CLIPPED mode has no effect since torus model not used.\n");
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(1254,errmsg, WARNING);
         }
         torus_display_mode = TORUS_CLIPPED_MODE;
         web.torus_body_flag = 0; web.torus_clip_flag = 1;
         update_display();
         break;

    case THICKEN_NODE: 
      thickenflag = (node->op1.toggle_state == ON_) ? 1 : 0;
         update_display();
         break;

    case METRIC_CONVERT_NODE:
         metric_convert_flag = (node->op1.toggle_state == ON_) ? 1 : 0;
         if ( metric_convert_flag )
             outstring("Converting form to vector using metric.\n");
         else outstring("Not using metric to convert form to vector.\n");
         break;

    case QUANTITIES_ONLY_NODE:
         old = quantities_only_flag;
         quantities_only_flag = (node->op1.toggle_state == ON_) ? 1 : 0;
         outstring("Using quantities only ");
         outstring(quantities_only_flag ? "ON.": "OFF.");
         outstring(old ? "(was ON)\n" : "(was OFF)\n");
         break;

    case SQUARED_GRADIENT_NODE:
         old = min_square_grad_flag;
         min_square_grad_flag = (node->op1.toggle_state == ON_) ? 1 : 0;
         outstring("Squared gradient minimization with Hessian ");
         outstring(min_square_grad_flag ? "ON.": "OFF.");
         outstring(old ? "(was ON)\n" : "(was OFF)\n");
         break;

    case LINEAR_METRIC_NODE:
         old = hessian_linear_metric_flag;
         hessian_linear_metric_flag = (node->op1.toggle_state == ON_) ? 1 : 0;
         outstring("Linear interpolation metric with Hessian ");
         outstring(hessian_linear_metric_flag ? "ON.": "OFF.");
         outstring(old ? "(was ON)\n" : "(was OFF)\n");
         break;

    case MKL_NODE:
#ifndef MKL
      kb_error(7761,"This Evolver not enabled for MKL.\n",RECOVERABLE);
#endif
      old = ysmp_flag;
      ysmp_flag = 
          (node->op1.toggle_state == ON_) ? MKL_FACTORING : MINDEG_FACTORING;
      change_hessian_functions(old,ysmp_flag);
      break;

    case YSMP_NODE: 
      old = ysmp_flag;
      ysmp_flag = 
          (node->op1.toggle_state == ON_) ? YSMP_FACTORING : MINDEG_FACTORING;
      change_hessian_functions(old,ysmp_flag);
      break;

    case METIS_FACTOR_NODE:
         old = ysmp_flag;
#ifndef METIS 
         kb_error(2055,"This Evolver not compiled with METIS.\n",RECOVERABLE);
#else
         ysmp_flag = METIS_FACTORING;
         change_hessian_functions(old,ysmp_flag);
#endif
         break;

    case BUNCH_KAUFMAN_NODE:
          flip_toggle(&BK_flag,node->op1.toggle_state,
                 "Bunch-Kaufman version of minimal degree");
             break;

    case SET_PROC_END_NODE:
    case SET_FUNC_END_NODE:
    case SET_PERM_PROC_END_NODE:
    case SHOW_END_NODE:
    case ELSE_NODE:
    case COND_ELSE_NODE:
         /* just a continue node */
         break;

    case MAKE_THREAD_LISTS_NODE:
         if ( !threadflag )
           break;
         if ( v_partition_stage_attr < 0 )
         { sprintf(errmsg,"make_thread_lists: stage and proc attributes don't exist.\n"); 
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
            kb_error(5822,errmsg,RECOVERABLE);
         }
         make_thread_lists();
         break;

    default:
            sprintf(errmsg,"Bad expression eval node type %d: %s.",
                node->type,tokname(node->type));
            if ( node->file_no >= 0 && node->file_no <= file_no_used )
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            else
              sprintf(errmsg+strlen(errmsg),"(appears to be corrupt node)\n");
            kb_error(1235,errmsg,RECOVERABLE);

            break;

  }

} /* end more_other_stuff() */


/********************************************************************
*
* function: get_toggle_value()
*
* purpose:  get boolean value of toggle for expression 
*/

int get_toggle_value(int tog  /* type of toggle */)
{
  switch (tog)
    {  case GEOMVIEW_NODE: return geomview_flag; 
       case VIEW_TRANSFORMS_USE_UNIQUE_NODE: 
             return view_transforms_unique_point_flag;
       case DETORUS_STICKY_NODE: return detorus_sticky;
       case BEZIER_BASIS_NODE: return bezier_flag;
       case SMOOTH_GRAPH_NODE: return smooth_graph_flag;
       case FULL_BOUNDING_BOX_NODE: return full_bounding_box_flag;
       case POP_DISJOIN_NODE: return pop_disjoin_flag;
       case POP_ENJOIN_NODE: return pop_enjoin_flag;
       case POP_TO_EDGE_NODE: return pop_to_edge_flag;
       case POP_TO_FACE_NODE: return pop_to_face_flag;
       case MPI_DEBUG_NODE: return mpi_debug;
       case MPI_LOCAL_BODIES_NODE: return mpi_local_bodies_flag; 
       case AMBIENT_PRESSURE_NODE: return web.pressure_flag;
       case ZENER_DRAG_NODE: return zener_drag_flag;
       case BACKCULL_NODE: return backcull_flag;
       case TORUS_FILLED_NODE: return web.full_flag;
       case VOLGRADS_EVERY_NODE: return volgrads_every_flag;
       case LOGFILE_NODE: return logfile_flag;
       case KEYLOGFILE_NODE: return logfile_flag;
       case LINEAR_NODE: return (web.modeltype == LINEAR);
       case QUADRATIC_NODE: return (web.modeltype == QUADRATIC);
       case LAGRANGE_NODE: return (web.modeltype == LAGRANGE);
       case KUSNER_NODE: return kusner_flag; 
       case ESTIMATE_NODE: return estimate_flag;
       case DETURCK_NODE: return unit_normal_flag;
       case HOMOTHETY_NODE: return web.homothety;
       case SLICE_VIEW_NODE: return slice_view_flag; 
       case CLIP_VIEW_NODE: return clip_view_flag; 
       case SQGAUSS_NODE: return sqgauss_flag; 
       case AUTOPOP_NODE: return autopop_flag;
       case IMMEDIATE_AUTOPOP_NODE: return immediate_autopop_flag;
       case AUTOPOP_QUARTIC_NODE: return autopop_quartic_flag;
       case AUTOCHOP_NODE: return autochop_flag;
       case QUIET_NODE: return quiet_flag; 
       case HESSIAN_QUIET_NODE: return hessian_quiet_flag; 
       case QUIETGO_NODE: return quiet_go_flag; 
       case QUIETLOAD_NODE: return quiet_load_flag; 
       case OLD_AREA_NODE: return old_area_flag;
       case APPROX_CURV_NODE: return approx_curve_flag;
       case RUNGE_KUTTA_NODE: return runge_kutta_flag; 
       case CHECK_INCREASE_NODE: return check_increase_flag; 
       case DEBUG_NODE: return yydebug; 
       case MEAN_CURV_NODE: return web.area_norm_flag; 
       case DIFFUSION_NODE: return web.diffusion_flag; 
       case GRAVITY_NODE: return web.gravflag; 
       case CONJ_GRAD_NODE: return conj_grad_flag; 
       case TRANSFORMS_NODE: return transforms_flag; 
       case CONF_EDGE_SQCURV_NODE: return conf_edge_curv_flag; 
       case EFFECTIVE_AREA_NODE: return effective_area_flag; 
       case RAW_CELLS_NODE: return !web.torus_body_flag && !web.torus_clip_flag; 
       case CONNECTED_CELLS_NODE: return web.torus_body_flag; 
       case CLIPPED_CELLS_NODE: return web.torus_clip_flag; 
       case THICKEN_NODE: return thickenflag; 
       case YSMP_NODE: return ysmp_flag; 
       case LINEAR_METRIC_NODE: return hessian_linear_metric_flag; 
       case METRIC_CONVERT_NODE: return metric_convert_flag; 
       case BUNCH_KAUFMAN_NODE: return BK_flag; 
       case SHOW_INNER_NODE: return innerflag; 
       case SHOW_OUTER_NODE: return outerflag; 
       case COLORMAP_NODE: return colorflag; 
       case HESSIAN_DIFF_NODE: return hessian_by_diff_flag; 
       case POST_PROJECT_NODE: return post_project_flag; 
       case MEAN_CURV_INT_NODE: return mean_curv_int_flag; 
       case OPTIMIZE_NODE: return !web.motion_flag; 
       case NORMAL_CURVATURE_NODE: return normal_curvature_flag; 
       case DIV_NORMAL_CURVATURE_NODE: return div_normal_curvature_flag; 
       case SHADING_NODE: return shading_flag; 
       case FACET_COLORS_NODE: return color_flag; 
       case BOUNDARY_CURVATURE_NODE: return boundary_curvature_flag; 
       case NORMAL_MOTION_NODE: return normal_motion_flag; 
       case PINNING_NODE: return check_pinning_flag; 
       case VIEW_4D_NODE: return view_4D_flag; 
       case MEMDEBUG_NODE: return memdebug; 
       case ITDEBUG_NODE: return itdebug; 
       case METRIC_CONVERSION_NODE: return metric_convert_flag; 
       case AUTORECALC_NODE: return autorecalc_flag; 
       case FORCE_POS_DEF_NODE: return make_pos_def_flag; 
       case GV_BINARY_NODE: return gv_binary_flag; 
       case SELF_SIMILAR_NODE: return self_similar_flag; 
       case AUTODISPLAY_NODE: return go_display_flag; 
       case RIBIERE_CG_NODE: return ribiere_flag; 
       case ASSUME_ORIENTED_NODE: return assume_oriented_flag; 
       case DIRICHLET_MODE_NODE: return dirichlet_flag; 
       case SOBOLEV_MODE_NODE: return sobolev_flag; 
       case KRAYNIKPOPVERTEX_FLAG_NODE: return kraynikpopvertex_flag; 
       case KRAYNIKPOPEDGE_FLAG_NODE: return kraynikpopedge_flag; 
       case K_ALTITUDE_FLAG_NODE: return K_altitude_flag; 
       case HESSIAN_NORMAL_NODE: return hessian_normal_flag; 
       case HESSIAN_SPECIAL_NORMAL_NODE: return hessian_special_normal_flag; 
       case HESSIAN_NORMAL_PERP_NODE: return hessian_normal_perp_flag; 
       case HESSIAN_NORMAL_ONE_NODE: return hessian_normal_one_flag; 
       case HESSIAN_DOUBLE_NORMAL_NODE: return hessian_double_normal_flag; 
       case INTERP_BDRY_PARAM_NODE: return interp_bdry_param; 
       case H_INVERSE_METRIC_NODE: return web.h_inverse_metric_flag; 
       case PSCOLORFLAG_NODE: return ps_colorflag;
       case PS_CMYKFLAG_NODE: return ps_cmykflag;
       case GRIDFLAG_NODE: return gridflag;
       case FORCE_EDGESWAP_NODE: return force_edgeswap_flag;
       case SEPTUM_FLAG_NODE: return septum_flag; 
       case BOX_FLAG_NODE: return box_flag;
       case SHOW_ALL_EDGES_NODE: return edgeshow_flag;
       case METIS_FACTOR_NODE: return ysmp_flag == METIS_FACTORING;
       case CROSSINGFLAG_NODE: return crossingflag;
       case LABELFLAG_NODE: return labelflag;
       case SHOW_ALL_QUANTITIES_NODE: return show_all_quantities;
       case QUANTITIES_ONLY_NODE: return quantities_only_flag;
       case VISIBILITY_TEST_NODE: return visibility_test;
       case SPARSE_CONSTRAINTS_NODE: return sparse_constraints_flag;
       case BLAS_FLAG_NODE: return blas_flag; 
       case BREAK_AFTER_WARNING_NODE : return break_after_warning;
       case BREAK_ON_WARNING_NODE : return break_on_warning;
       case AUGMENTED_HESSIAN_NODE: return augmented_hessian_flag;
       case VERBOSE_NODE: return verbose_flag;
       case FUNCTION_QUANTITY_SPARSE_NODE: return quantity_function_sparse_flag;
       case BIG_ENDIAN_NODE: return big_endian_flag; 
       case LITTLE_ENDIAN_NODE : return little_endian_flag;
       case RGB_COLORS_FLAG_NODE : return rgb_colors_flag;
       default:
            sprintf(errmsg,"Internal error:  Toggle value omitted for toggle %d.\n",tog);
            kb_error(1236,errmsg,WARNING);
    }
    return 0;
} /* end get_toggle_value */

/*************************************************************************
*
* function: get_internal_variable()
*
* purpose: get value of various variables visible to user.
*/

REAL get_internal_variable(int vartok /* token number of variable */)
{ 
  switch(vartok)
  { 
    case V_DETORUS_EPSILON: return dt_eps; 
    case V_BOUNDING_BOX_COLOR: return bounding_box_color;
    case V_HIGH_BOUNDARY: return web.highbdry;
    case V_HIGH_CONSTRAINT: return web.highcon;
    case V_RANDOM: return kb_drand();
    case V_MPI_MAXTASK: 
#ifdef MPI_EVOLVER
         return mpi_nprocs-1;
#else
         return 1;
#endif
#ifdef MPI_EVOLVER
    case V_CORONA_STATE: 
      return (REAL)mpi_corona_state; 
#endif
    case V_VERTEXCOUNT:
      return (REAL)web.skel[VERTEX].count; 
    case V_EDGECOUNT:
      return (REAL)web.skel[EDGE].count; 
    case V_FACETCOUNT:
      return (REAL)web.skel[FACET].count; 
    case V_BODYCOUNT:
      return (REAL)web.skel[BODY].count; 
    case V_FACETEDGECOUNT:
      return (REAL)web.skel[FACETEDGE].count; 
    case V_ENERGY:
      return web.total_energy; 
    case V_GAP_CONSTANT:
      return web.spring_constant; 
    case V_TIME:
      return total_time; 
    case V_JIG_TEMP:
      return web.temperature; 
    case V_AREA:
    case V_LENGTH:
      return web.total_area; 
    case V_ITER_COUNTER:
      return (REAL)gocount; 
    case V_AMBIENT_PRESSURE:
      return web.pressure; 
    case V_DIFFUSION:
      return web.diffusion_const; 
    case V_SCALE:
      return web.scale; 
    case V_SCALE_SCALE:
      return web.scale_scale; 
    case V_SPACE_DIMENSION:
      return (REAL)SDIM; 
    case V_SURFACE_DIMENSION:
      return (REAL)web.dimension; 
    case V_TORUS:
      return (REAL)web.torus_flag; 
    case V_SYMMETRY_GROUP:
      return (REAL)web.symmetry_flag; 
    case V_SIMPLEX:
      return (REAL)(web.representation == SIMPLEX); 
    case V_INTEGRAL_ORDER:
      return ((REAL)web.gauss1D_order+2)/2; 
    case V_INTEGRAL_ORDER_1D:
      return (REAL)web.gauss1D_order; 
    case V_INTEGRAL_ORDER_2D:
      return (REAL)web.gauss2D_order; 
    case V_TOLERANCE:
      return web.tolerance; 
    case V_TARGET_TOLERANCE:
      return web.target_tolerance; 
    case V_THICKNESS:
      return thickness; 
    case V_HESSIAN_SLANT_CUTOFF:
      return hessian_slant_cutoff; 
    case V_HESS_EPSILON:
      return hessian_epsilon; 
    case GRAV_CONST_NODE:
      return web.grav_const; 
    case V_EQUI_COUNT:
      return (REAL)web.equi_count; 
    case V_EDGESWAP_COUNT:
      return (REAL)web.edgeswap_count; 
    case V_T1_EDGESWAP_COUNT:
      return (REAL)web.t1_edgeswap_count; 
    case V_DELETE_COUNT:
      return (REAL)web.edge_delete_count + web.facet_delete_count; 
    case V_EDGE_DELETE_COUNT:
      return (REAL)web.edge_delete_count; 
    case V_FACET_DELETE_COUNT:
      return (REAL)web.facet_delete_count; 
    case V_REFINE_COUNT:
      return (REAL)web.edge_refine_count + web.facet_refine_count;; 
    case V_EDGE_REFINE_COUNT:
      return (REAL)web.edge_refine_count; 
    case V_FACET_REFINE_COUNT:
      return (REAL)web.facet_refine_count; 
    case V_NOTCH_COUNT:
      return (REAL)web.notch_count; 
    case V_DISSOLVE_COUNT:
      return (REAL)web.vertex_dissolve_count + web.edge_dissolve_count
             + web.facet_dissolve_count + web.body_dissolve_count; 
    case V_VERTEX_DISSOLVE_COUNT:
      return (REAL)web.vertex_dissolve_count; 
    case V_EDGE_DISSOLVE_COUNT:
      return (REAL)web.edge_dissolve_count; 
    case V_FACET_DISSOLVE_COUNT:
      return (REAL)web.facet_dissolve_count; 
    case V_BODY_DISSOLVE_COUNT:
      return (REAL)web.body_dissolve_count; 
    case V_EDGE_REVERSE_COUNT:
      return (REAL)web.edge_reverse_count; 
    case V_FACET_REVERSE_COUNT:
      return (REAL)web.facet_reverse_count; 
    case V_FIX_COUNT:
      return web.fix_count;
    case V_THIS_TASK:
      return this_task;
    case V_WINDOW_ASPECT_RATIO:
      return window_aspect_ratio;
    case V_STRING_CURVE_TOLERANCE:
      return string_curve_tolerance; 
    case V_MINDEG_DEBUG_LEVEL:
      return mindeg_debug_level;
    case V_MINDEG_MARGIN:
      return mindeg_margin; 
    case V_MINDEG_MIN_REGION_SIZE:
      return mindeg_min_region_size; 
    case V_UNFIX_COUNT:
      return web.unfix_count;
    case V_POP_COUNT:
      return web.vertex_pop_count + web.edge_pop_count;
    case V_VERTEX_POP_COUNT:
      return web.vertex_pop_count;
    case V_EDGE_POP_COUNT:
      return web.edge_pop_count;
    case V_POP_TRI_TO_EDGE_COUNT:
      return web.pop_tri_to_edge_count;
    case V_POP_EDGE_TO_TRI_COUNT:
      return web.pop_edge_to_tri_count;
    case V_POP_QUAD_TO_QUAD_COUNT:
      return web.pop_quad_to_quad_count;
    case V_WHERE_COUNT:
      return (REAL)web.where_count; 
    case V_EIGENPOS:
      return (REAL)eigen_pos; 
    case V_EIGENNEG:
      return (REAL)eigen_neg; 
    case V_EIGENZERO:
      return (REAL)eigen_zero; 
    case V_CHECK_COUNT:
      return (REAL)check_count; 
    case V_BAD_NEXT_PREV_COUNT:
      return (REAL)bad_next_prev_count; 
    case V_INCONSISTENT_BODIES_COUNT:
      return (REAL)inconsistent_bodies_count; 
    case V_EDGE_LOOP_COUNT:
      return (REAL)edge_loop_count; 
    case V_EDGES_SAME_VERTICES_COUNT:
      return (REAL)edges_same_vertices_count; 
    case V_FACETS_SAME_VERTICES_COUNT:
      return (REAL)facets_same_vertices_count; 
    case V_BAD_ERRORS_COUNT:
      return (REAL)bad_errors_count; 
    case V_VISIBILITY_DEBUG:
      return (REAL)visdebuglevel; 
    case V_SCROLLBUFFERSIZE:
      return (REAL)scrollbuffersize;
    case V_BREAKFLAG:
      return (REAL)breakflag; 
    case V_PICKVNUM:
      return (REAL)pickvnum; 
    case V_PICKENUM:
      return (REAL)pickenum; 
    case V_PICKFNUM:
      return (REAL)pickfnum; 
    case V_LINEAR_METRIC_MIX:
      return linear_metric_mix; 
    case V_QUADRATIC_METRIC_MIX:
      return quadratic_metric_mix; 
    case V_RANDOM_SEED:
      return (REAL)random_seed; 
    case V_LAST_EIGENVALUE:
      return last_eigenvalue; 
    case V_LAST_HESSIAN_SCALE:
      return last_hessian_scale; 
    case V_LAGRANGE_ORDER:
      return (REAL)web.lagrange_order; 
    case V_SCALE_LIMIT:
      return web.maxscale; 
    case V_BRIGHTNESS:
      return brightness; 
    case V_LAST_ERROR: return last_error;
    case V_BACKGROUND: return background_color;
    case V_PS_LABELSIZE: return ps_labelsize;
    case V_PS_STRINGWIDTH: return ps_stringwidth;
    case V_PS_FIXEDEDGEWIDTH: return ps_fixededgewidth;
    case V_PS_TRIPLEEDGEWIDTH: return ps_tripleedgewidth;
    case V_PS_CONEDGEWIDTH: return ps_conedgewidth;
    case V_PS_BAREEDGEWIDTH: return ps_bareedgewidth;
    case V_PS_GRIDEDGEWIDTH: return ps_gridedgewidth;
    case V_EVERYTHING_QUANTITIES: return everything_quantities_flag;
    case V_AUTOCHOP_LENGTH: return autochop_length;
    case GRAV_CONST_TOK: return web.grav_const;
    case V_TRANSFORM_COUNT:
      return transform_count == 0 ? 1 : transform_count;
    case V_CLOCK:
#if defined (__SYS_TIMES_H__) && defined(HZ)
      { struct tms t;
        times(&t);
        return t.tms_utime/(REAL)HZ;
      }
#else
#if defined (_SYS_TIMES_H) && defined(CLK_TCK)
      { struct tms t;    /* SGI, anyway */
        times(&t);
        return t.tms_utime/(REAL)CLK_TCK;
      }
#else
#ifdef CLOCKS_PER_SEC
      return clock()/(REAL)CLOCKS_PER_SEC;
#else
      return 0.0;
#endif
#endif
#endif
      break;
    
    case V_CPU_COUNTER:
      { long long int cycles; 
        PROF_NOW(cycles);
        return (REAL)PROF_CYCLES(cycles);
      }

    case V_MEMARENA:
#if defined(M_MXFAST) && defined(IRIS)
      { struct mallinfo m = mallinfo();
        return (REAL)m.arena;
      }
#else
#if defined(_UNISTD_H)
  /* do this only on unix systems with unistd.h */
  return  (char*)sbrk(0)-(char*)&evolver_version;
#else
#ifdef MSC
      { 

          /*
        struct _heapinfo hinfo;
        size_t mem_use=0,mem_free=0;
        hinfo._pentry = NULL;
        while ( _heapwalk(&hinfo) == _HEAPOK )
           if (hinfo._useflag) { mem_use+= hinfo._size; }
           else {  mem_free += hinfo._size; }
        return (REAL)(mem_use+mem_free);
        */
        PROCESS_MEMORY_COUNTERS meminfo;
        meminfo.cb = sizeof(meminfo);
        GetProcessMemoryInfo(GetCurrentProcess(),&meminfo,sizeof(meminfo));
        return (REAL)meminfo.WorkingSetSize;
      }          

#else
#ifdef TC
      { struct heapinfo hinfo;
        long mem_use=0,mem_free=0;
        hinfo.ptr = NULL;
        while ( heapwalk(&hinfo) == _HEAPOK )
           if (hinfo.in_use) { mem_use+= hinfo.size; }
           else {  mem_free += hinfo.size; }
        return (REAL)(mem_use+mem_free);
      }          

#else
      return 0.0;
#endif
#endif
#endif
#endif
      break;

    case V_MEMUSED:
#if defined(M_MXFAST) && defined(IRIS)
      { struct mallinfo m = mallinfo();
        return (REAL)(m.uordblks + m.usmblks);
      }
#else
#ifdef WIN32
      { struct _heapinfo hinfo;
        size_t mem_use=0,mem_free=0;
        hinfo._pentry = NULL;
        while ( _heapwalk(&hinfo) == _HEAPOK )
           if (hinfo._useflag) { mem_use+= hinfo._size; }
           else {  mem_free += hinfo._size; }
        return (REAL)(mem_use);
      }          

#else
      return 0.0;
#endif
#endif
      break;

    default: 
      sprintf(errmsg,"Internal: illegal variable number %d.\n",vartok);
      kb_error(1208, errmsg,RECOVERABLE);

  }
  return 0.0;

} // end get_internal_variable()

/**********************************************************************
*
*  function: tree_copy()
*
*  purpose: copy tree in linear postorder form
*              allocates space for tree in destination
*
*/

void tree_copy(
  struct expnode *dest,
  struct treenode *src
)
{
  struct treenode *enode;
  size_t count,n;

  if ( dest == NULL ) return;
  if ( dest->start ) myfree((char*)dest->start);
  if ( src == NULL )  
  { dest->start = dest->root = NULL; return; }
  enode=src; 
  while ( enode->left || enode->right ) 
  { if ( enode->left < enode->right ) enode+=enode->left;
    else enode += enode->right;
  }
  count = src - enode + 1;
  dest->start = (struct treenode *)mycalloc(count+3,sizeof(struct treenode));
  dest->flag |= USERCOPY;
  memcpy((char*)(dest->start+2),(char*)enode,count*sizeof(struct treenode));
  dest->root = dest->start + count + 1;
  dest->start[1].type = SETUP_FRAME_NODE;
  /* make copies of strings and locallists */
  for ( enode=dest->start+2, n = 0 ; n < count ; enode++,n++ )
  { if ( enode->flags & HAS_STRING )
    { char *s = mycalloc(strlen(enode->op1.string)+1,sizeof(char));
      strcpy(s,enode->op1.string);
      enode->op1.string = s;
    }
    if ( enode->flags & HAS_STRING_5 )
    { char *s = mycalloc(strlen(enode->op5.string)+1,sizeof(char));
      strcpy(s,enode->op5.string);
      enode->op5.string = s;
    }
    if ( enode->flags & HAS_LOCALLIST )
    { struct locallist_t *locals =
         (struct locallist_t *)mycalloc(1,sizeof(struct locallist_t));
      *locals = *enode->op5.locals;
      locals->list = (struct localvar_t *)mycalloc(enode->op5.locals->count,
                             sizeof(struct localvar_t));
      memcpy(locals->list,enode->op5.locals->list,enode->op5.locals->count*
                              sizeof(struct localvar_t));
      enode->op5.locals = locals;
    }
  }
  /* copy root to first place */
  dest->start[0] = *src;
  if ( dest->start[0].left ) dest->start[0].left += (int)count+1;
  if ( dest->start[0].right ) dest->start[0].right += (int)count+1;

  /* FINISH node */
  dest->start[count+2].type = FINISHED_NODE;

  stack_usage(dest);
}  /* end tree_copy() */


/**********************************************************************
*
*  function: perm_tree_copy()
*
*  purpose: copy tree in linear postorder form
*              allocates permanent space for tree in destination
*
*/

void perm_tree_copy(
  struct expnode *dest,
  struct treenode *src
)
{
  struct treenode *enode;
  size_t count,n;
  if ( dest == NULL ) return;
  if ( dest->start ) free((char*)dest->start);
  if ( src == NULL )  
  { dest->start = dest->root = NULL; return; }
  enode=src; 
  while ( enode->left || enode->right ) 
  { if ( enode->left < enode->right ) enode+=enode->left;
    else enode += enode->right;
  }
  count = src - enode + 1;
  dest->start = (struct treenode *)calloc(count+3,sizeof(struct treenode));
  dest->flag |= USERCOPY;
  memcpy((char*)(dest->start+2),(char*)enode,count*sizeof(struct treenode));
  dest->root = dest->start + count + 1;
  dest->start[1].type = SETUP_FRAME_NODE;
  /* make copies of strings */
  for ( enode=dest->start+2, n = 0 ; n < count ; enode++,n++ )
  { if ( enode->flags & HAS_STRING )
    { char *s = calloc(strlen(enode->op1.string)+1,sizeof(char));
      strcpy(s,enode->op1.string);
      enode->op1.string = s;
    }
    enode->flags |= PERMNODE;
  }
  /* copy root to first place */
  dest->start[0] = *src;
  if ( dest->start[0].left ) dest->start[0].left += (int)count + 1;
  if ( dest->start[0].right ) dest->start[0].right += (int)count + 1;
  dest->start[0].flags |= PERMNODE;

  /* FINISH node */
  dest->start[count+2].type = FINISHED_NODE;

  stack_usage(dest);

} // end perm_tree_copy()


/*******************************************************************
*
* function: reduce_strings()
*
* purpose:  converts escape sequences to ASCII in place  
*
*/
void reduce_string(char *s)
{ char *c = s;
  while ( *s ) /* reduce escape sequences */
  if ( *s == '\\' )
  { 
    switch ( s[1] )
    { case 'n' : *(c++) = '\n'; s += 2; break;
      case 't' : *(c++) = '\t'; s += 2; break;
      case 'b' : *(c++) = '\b'; s += 2; break;
      case 'r' : *(c++) = '\r'; s += 2; break;
      case '\\' : *(c++) = '\\'; s += 2; break;
      case 'q' : *(c++) = '"'; s += 2; break;
      default:  *(c++) =  s[1]; s += 2; break;
    }
  }
  else if (*s == '\n' ) s++;  /* omit newlines */
  else *(c++) = *(s++);
  *c = '\0';

} // end reduce_string()

/***********************************************************************
*
* function: set_body()
*
* purpose: set facet body, or edge body in string model.
*          Can set to NULLID.
*/

void set_body(
  element_id id, /* edge or facet, inverted for backbody */
  body_id b_id
)
{
  if ( id_type(id) == FACET )
  {
    set_facet_body(id,b_id);

    return;
  }

  /* STRING model only hereafter! */

  if ( (id_type(id) != EDGE) || (web.representation != STRING) )
  { sprintf(errmsg,"Trying to set body of wrong type element.\n");
    kb_error(2056,errmsg,RECOVERABLE);
  }
  if ( !valid_id(b_id) )
  { facetedge_id fe,fe_id,next_fe,prev_fe;
    facet_id f_id;
    body_id bb_id;

    /* unsetting edge body, so dissolving facetedge */

    fe_id = get_edge_fe(id);
    f_id = get_fe_facet(fe_id);
    if ( !valid_id(f_id) ) return;
    if ( inverted(f_id) ) 
    { fe_id = get_next_facet(fe_id);
      f_id = get_fe_facet(fe_id);
    } 
    if ( !valid_id(f_id) ) return;
    if ( inverted(f_id) ) return; 

    /* check for being at beginning or end of edge arc */
    fe = fe_id;
    if ( valid_id(get_next_edge(fe)) && valid_id(get_prev_edge(fe)) )
    { /* check for full loop */
      do { fe = get_next_edge(fe); }
      while ( valid_id(fe) && !equal_id(fe,fe_id) );
      if ( !valid_id(fe) )
      { if ( verbose_flag )
        { sprintf(msg,"Not unbodying edge %s since would make facet two arcs.\n",
            ELNAME(id));
          outstring(msg);
        }
        return;
      }
    }
    /* Okay, ready to party. */
    if ( everything_quantities_flag )
    { bb_id = get_facet_body(f_id);
      if ( valid_id(bb_id)  )
      { unapply_method(id,get_body_volmeth(bb_id));
      }
    }
    next_fe = get_next_edge(fe_id);
    prev_fe = get_prev_edge(fe_id);
    if ( valid_id(next_fe) ) 
    { set_prev_edge(next_fe,NULLID);
      set_facet_fe(f_id,next_fe);
    }
    if ( valid_id(prev_fe) )
      set_next_edge(prev_fe,NULLID);
    else if ( !valid_id(next_fe) ) 
      set_facet_fe(f_id,NULLID);
    if ( valid_id(get_next_facet(fe_id)) )
      set_edge_fe(id,get_next_facet(fe_id));
    set_next_facet(get_prev_facet(fe_id),get_next_facet(fe_id));
    set_prev_facet(get_next_facet(fe_id),get_prev_facet(fe_id));
    free_element(fe_id);
  }
  else /* valid id */
  { /* string model */
    facet_id f_id = get_body_facet(b_id);
    facetedge_id efe,eefe,fe,start_fe;
    facetedge_id newfe;
    body_id bb_id;
    int inserted_flag=0;  /* whether a linkup made */

    if ( !valid_id(f_id) )
    { sprintf(errmsg,
            "Need to set the body's facet before assigning edges.\n");
      kb_error(2057,errmsg,RECOVERABLE);
    }
    if ( everything_quantities_flag )
    { bb_id = get_facet_body(f_id);
      if ( valid_id(bb_id) )
      { unapply_method(id,get_body_volmeth(bb_id));
      }
      apply_method_num(id,get_body_volmeth(b_id));
    }
    newfe = new_facetedge(f_id,id);
    efe = get_edge_fe(id);
    if ( valid_id(get_fe_facet(efe)) )
    { eefe = get_next_facet(efe);
      set_next_facet(efe,newfe);
      set_next_facet(newfe,eefe);
      set_prev_facet(eefe,newfe);
      set_prev_facet(newfe,efe);
    }
    else
    { if (valid_id(efe)) free_element(efe);
      set_next_facet(newfe,newfe);
      set_prev_facet(newfe,newfe);
      set_edge_fe(id,newfe);
    }
    /* see if can place at head or end of chain */
    fe = start_fe = get_facet_fe(f_id);
    if ( !valid_id(fe) )
    { /* naked facet */
      set_facet_fe(f_id,newfe); inserted_flag = 1;
    }
    else
    { if ( !valid_id(get_prev_edge(fe)) )
      { if ( equal_id(get_fe_tailv(fe),get_edge_headv(id)) )
        { set_next_edge(newfe,fe); set_prev_edge(fe,newfe); 
          set_facet_fe(f_id,newfe); inserted_flag = 1;
        }
      }
      while ( valid_id(get_next_edge(fe)) &&
                        !equal_id(get_next_edge(fe),start_fe) )
           fe = get_next_edge(fe);
      if ( !valid_id(get_next_edge(fe)) )
      { if ( equal_id(get_fe_headv(fe),get_edge_tailv(id)) )
        { set_next_edge(fe,newfe); set_prev_edge(newfe,fe); 
          inserted_flag = 1;
        }
      }
    }
    if ( !inserted_flag ) 
    { sprintf(errmsg,"Could not link edge %s onto facet %s chain.\n",
           ELNAME(id),ELNAME1(f_id));
      kb_error(2058,errmsg,WARNING);
    }
  }
} /* end set_body() */

/***************************************************************************
*
* function: change_hessian_functions()
*
* purpose: Switch sets of Hessian functions.
*/
void change_hessian_functions(
  int oldh,
  int newh
  )
{
  if ( oldh == newh ) return;
  switch ( newh )
  { case MINDEG_FACTORING:
      {
         sp_mul_func = bk_mul;
         sp_AIJ_setup_func= bk_AIJ_setup;
         sp_constraint_setup_func = bk_constraint_setup;
         sp_hess_project_setup_func= BK_hess_project_setup;
         sp_factor_func = xmd_factor;
         sp_solve_func = xmd_solve;
         sp_solve_multi_func = xmd_solve_multi;
         sp_ordering_func = NULL;
         sp_CHinvC_func = sp_CHinvC;
         outstring("Using alternate minimal degree. ");
         switch ( oldh ) 
         {  case YSMP_FACTORING: outstring("(was YSMP)\n"); break;
            case MINDEG_FACTORING: outstring("(was alt min deg)\n"); break;
            case METIS_FACTORING: outstring("(was metis)\n"); break;
            case MKL_FACTORING: outstring("(was MKL)\n"); break;
         }
      }
      break;
     case YSMP_FACTORING:
      { 
         sp_mul_func = bk_mul;
         sp_AIJ_setup_func= bk_AIJ_setup;
         sp_constraint_setup_func = bk_constraint_setup;
         sp_hess_project_setup_func= BK_hess_project_setup;
         sp_factor_func = ysmp_factor;
         sp_CHinvC_func = sp_CHinvC;
         sp_solve_func = ysmp_solve;
         sp_solve_multi_func = ysmp_solve_multi;
         sp_ordering_func = NULL;
         outstring("Using YSMP. ");
         switch ( oldh ) 
         {  case YSMP_FACTORING: outstring("(was YSMP)\n"); break;
            case MINDEG_FACTORING: outstring("(was alt min deg)\n"); break;
            case METIS_FACTORING: outstring("(was metis)\n"); break;
            case MKL_FACTORING: outstring("(was MKL)\n"); break;
         }
      }
      break;
    case METIS_FACTORING:
     {
         sp_mul_func = bk_mul;
         sp_AIJ_setup_func= bk_AIJ_setup;
         sp_constraint_setup_func = bk_constraint_setup;
         sp_hess_project_setup_func= BK_hess_project_setup;
         sp_factor_func = ysmp_factor;
         sp_CHinvC_func = sp_CHinvC;
         sp_solve_func = ysmp_solve;
         sp_solve_multi_func = ysmp_solve_multi;
/*
tree stuff not working
         sp_factor_func = tree_factor;
         sp_solve_func = tree_solve;
         sp_solve_multi_func = tree_solve_multi;
*/
         sp_ordering_func = metis_order;
         outstring("Using Metis ordering in ysmp factoring. ");
         switch ( oldh ) 
         {  case YSMP_FACTORING: outstring("(was YSMP)\n"); break;
            case MINDEG_FACTORING: outstring("(was alt min deg)\n"); break;
            case METIS_FACTORING: outstring("(was metis)\n"); break;
            case MKL_FACTORING: outstring("(was MKL)\n"); break;
         }
     }
     case MKL_FACTORING:
      { 
         sp_mul_func = bk_mul;
         sp_AIJ_setup_func= bk_AIJ_setup;
         sp_constraint_setup_func = bk_constraint_setup;
         sp_hess_project_setup_func= BK_hess_project_setup;
         sp_factor_func = mkl_factor;
         sp_CHinvC_func = sp_CHinvC;
         sp_solve_func = mkl_solve;
         sp_solve_multi_func = mkl_solve_multi;
         sp_ordering_func = NULL;
         outstring("Using MKL. ");
         switch ( oldh ) 
         {  case YSMP_FACTORING: outstring("(was YSMP)\n"); break;
            case MINDEG_FACTORING: outstring("(was alt min deg)\n"); break;
            case METIS_FACTORING: outstring("(was metis)\n"); break;
            case MKL_FACTORING: outstring("(was MKL)\n"); break;
         }
      }
      break;
 
  }
  ysmp_flag = newh;
}  /* end change_hessian_functions() */
