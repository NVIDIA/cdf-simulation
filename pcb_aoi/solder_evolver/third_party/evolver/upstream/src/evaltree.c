/************************************************************
*  This file is part of the Surface Evolver source code.    *
*  Programmer:  Ken Brakke, brakke@susqu.edu                *
************************************************************/


/*****************************************************************
*
*  File: evaltree.c
*
*  Purpose: To execute expression and command trees.
*          Tree nodes are stored in postorder linear form
*          for fast postorder execution.
*    
*/

#include "include.h" 
#include "ytab.h"
 
/* for breakpoints */
#define BREAKMAX 100
struct breakinfo { int name_id; int line; } breaklist[BREAKMAX];
int breakcount;
struct eval_frame *subshell_frame[100];

#ifdef PROF_EVALS
#define PROF_EVAL_PUSH_AND_SWITCH(new_proc_ptr)  \
          PROF_EVAL_END(ex_profiling);   \
          *(struct expnode**)(++stacktop) = ex_profiling;  \
          ex_profiling = new_proc_ptr;  \
          PROF_EVAL_START(ex_profiling); \
          PROF_COUNT_INCR(ex_profiling);
#define PROF_EVAL_POP   PROF_EVAL_END(ex_profiling);  \
                        ex_profiling = *(struct expnode**)(stacktop--); \
                        PROF_EVAL_START(ex_profiling);
#define PROF_SPACE 1
#else
#define PROF_EVAL_PUSH_AND_SWITCH(new_proc_ptr)
#define PROF_EVAL_POP
#define PROF_SPACE 0
#endif



/*****************************************************************
*
*  Function eval()
*
*  Purpose: runtime evaluation of expressions and commands.
*    The big switch statement is split between two functions to
*    get functions small enough for DOS compilers. All nodes
*    manipulating the stack are in the eval() function, and 
*    others in other_stuff().  This should not impose any speed
*    penalty on expressions.
*
*  Notes: The evaluation stack is a local variable, to permit
*  parallel evaluations on shared memory machines.  To prevent
*  eval() from getting too long, many cases have been moved
*  to evalmore(), which is called by the default case of eval().
*  But for efficiency, most frequently evaluated cases should
*  be in eval().
*  
*  Uses one permanent stack per thread.  Recursive calls to eval()
*  wind up using same stack (but progressively, of course).
*/

int current_debug_line;
int debugging_flag;

/* Number of stack slots occupied by a frame structure */
#define FRAME_SPACE ((int)((sizeof(struct eval_frame)+sizeof(REAL)-1)/sizeof(REAL)))

// kludge due to localstack being offset in this file and pointer in others
#undef get_localp
#define get_localp(id) (newstack + localstack + localbase->list[(id)&GLOBMASK].offset)

REAL eval (
  struct expnode *ex_original,   /* expression tree */
  REAL *params,         /* vector of parameters */
  element_id self_id,   /* reference element, if any */
  struct eval_frame *parent_frame  /* not used anymore */
)
{ 
  struct treenode *node; /* currently executing node */

  struct thread_data *td = GET_THREAD_DATA;
  #define newstack  (td->eval_stack)
  #define stackmax  (td->eval_stack_size)
  #define stacktop  (td->stack_top)
  #define this_frame ((struct eval_frame*)(newstack + td->frame_spot))

  int entry_flag = BASE_OF_EVAL; /* so can set frame flag marking entry to eval() */
  
  /* variables that act like registers for stack gymnastics */
  struct expnode ex_current = *ex_original;

  struct locallist_t *localbase;
  int localcount;
  element_id q_id = self_id;  /* innermost loop element */
  REAL return_value;
  struct treenode *return_node = NULL;
  int localstack; /* offset of base of local stack, after frame structure */

  /* miscellaneous local variables useful in executing nodes */
  int k,n,i;
  REAL x,y;
  element_id id; 
  REAL *bins;
  REAL hi;
  REAL lo;
  REAL val;
  int old_flag = iterate_flag;
  facet_id f_id;
  REAL vect[MAXCOORD];
  facetedge_id fe;
  vertex_id v_id;
  int recalc_flag = 0;
  int update_display_flag = 0;
  struct boundary *bdry;
  REAL *histo_data;
  int histo_max;
  int histo_count;
  struct global *g;
  int oldquiet;
#ifdef PROF_EVALS
  struct expnode *ex_profiling = ex_original; // purely for profiling
#endif
  
  if ( ex_original == NULL ) return 0.0;
  ex_current = *ex_original;

  if ( ex_current.start == NULL )
  { sprintf(errmsg,"Trying to evaluate null expression for %s.\n",ex_current.name);
    kb_error(1253,errmsg,WARNING);
    return 0.0;
  }

  PROF_EVAL_START(ex_profiling);
  PROF_COUNT_INCR(ex_profiling);

  if ( !breakflag ) iterate_flag = 2; /* for interrupt handler */

  if ( ex_current.start[1].type != SETUP_FRAME_NODE )
  { sprintf(errmsg,"Internal error in eval(): no frame setup\n");
    sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[ex_current.start[1].file_no],ex_current.start[1].line_no);
    kb_error(3987,errmsg,RECOVERABLE);
  }
 
  for ( node = ex_current.start+1 ; ; node++ )
  {
    if ( single_step_debugging && (node->line_no != current_debug_line) )
    { char prompt_string[100];
      current_debug_line = node->line_no;
      subshell_depth++;
      subshell_frame[subshell_depth] = this_frame;
      setjmp(jumpbuf[subshell_depth]);
      if ( subshell_depth == 1 )
        sprintf(prompt_string,"Debug (\"%s\" line %d): ",ex_current.name,node->line_no);
      else
      { sprintf(prompt_string,"Debug command(%d): ",subshell_depth);
      }
      /* command read and execute loop */
      debugging_flag = 1;
      single_step_debugging = 0;
      exec_commands(NULL,prompt_string);
      subshell_depth--;
      debugging_flag = 0;
    }  
    else
    /* have reached node set to a breakpoint */
    if ( node->flags & BREAKPOINT_NODE )
    { char prompt_string[100];
      current_debug_line = node->line_no;
      subshell_depth++;
      subshell_frame[subshell_depth] = this_frame;
      setjmp(jumpbuf[subshell_depth]);
      if ( subshell_depth == 1 )
        sprintf(prompt_string,"Debug (\"%s\" line %d): ",ex_current.name,node->line_no);
      else
      { sprintf(prompt_string,"Debug command(%d): ",subshell_depth);
      }
      /* command read and execute loop */
      debugging_flag = 1;
      exec_commands(NULL,prompt_string);
      subshell_depth--;
      debugging_flag = 0;
    }

//mem_sanity_check();

    switch ( node->type )
    {
      struct treenode *where,*enode,*eroot;

      case SETUP_FRAME_NODE: /* first node of any procedure */
        { int parent_frame_spot;
          int stackused = (int)(stacktop-newstack);
          
          /* check stack space */
          localbase = ex_current.locals;
          if ( ex_current.locals )
            localcount = ex_current.locals->totalsize;
          else localcount = 0;
          if ( stackmax  <  stackused + localcount + ex_current.stack_max + FRAME_SPACE + 20 ) 
          { 
            stackmax = stackused + localcount + ex_current.stack_max + FRAME_SPACE + 300;
            newstack = (REAL*)realloc(newstack,stackmax*sizeof(REAL) );
            stacktop = newstack + stackused;
            newstack[stackmax-1]  = STACKMAGIC;  /* sentinel */         
          } 

          /* find parent frame, if any */
          parent_frame_spot = td->frame_spot;
          if ( stacktop == newstack  )
             parent_frame = NULL;

          else
             parent_frame = (struct eval_frame*)(newstack + parent_frame_spot);
          stacktop++;

          /* Set up first frame at current stack position */
          td->frame_spot = (int)(stacktop - newstack);  /* sets this_frame by macro */
          stacktop += FRAME_SPACE-1;
          this_frame->base_ex = ex_current;
          this_frame->basenode = &node;  
          this_frame->parent_frame_spot = parent_frame_spot;
          td->frame_spot = (int)((REAL*)this_frame - newstack);  // redundant?
          this_frame->self_id = self_id;
          this_frame->return_node = return_node;
          this_frame->flags = entry_flag;
          if ( stackused == 0 )
            this_frame->flags |= BASE_OF_WHOLE_STACK;
          entry_flag = 0;
          if ( parent_frame ) 
          { this_frame->flags |= parent_frame->flags  & IN_ELEMENT_LOOP;
            if ( return_node )
                this_frame->flags |= return_node->flags & IN_ELEMENT_LOOP;
          }   
          localstack = (int)((stacktop + 1) - newstack);
          
          /* local variables after frame */
          if ( localcount )
          { stacktop++;
            memset((char*)stacktop,0,localcount*sizeof(REAL));
            stacktop += localcount-1; /* room for local variables */
          }

        }
        break;  /* end SETUP_FRAME_NODE */

      case ABORT_NODE:
/*
          if ( subshell_depth )
          { exit_flag = 1;
            break;
          }
*/
          breakflag = BREAKABORT;
          break;

      case SUBCOMMAND_NODE:
    { char prompt_string[100];
      char *pmpt;
      int old_trace_spot = eval_stack_trace_spot;
      REAL *subshell_stack;
      subshell_depth++;
      subshell_frame[subshell_depth] = this_frame;
      subshell_stack = stacktop;
      setjmp(jumpbuf[subshell_depth]);
      stacktop = subshell_stack;  // set stack properly on return from error
      if ( subshell_depth == 1 )
       pmpt = "Subcommand: ";
      else
      { sprintf(prompt_string,"Subcommand(%d): ",subshell_depth);
        pmpt = prompt_string;
      }
      PUSH_TRACE;
      /* command read and execute loop */
      exec_commands(NULL,pmpt);
      subshell_depth--;
      eval_stack_trace_spot = old_trace_spot;
      break;
    }

      case SET_BREAKPOINT_NODE:
    { int breakline = (int)(*stacktop--);
      struct expnode *proc;
      struct treenode *nodespot;
      int found = 0;
      /* now set flag bit in first node on line */
      proc =  &(globals(node->op1.name_id)->value.proc); 
      for ( nodespot = proc->start+1 ; nodespot != proc->root ; nodespot++ )
        if ( nodespot->line_no == breakline )
        { nodespot->flags |= BREAKPOINT_NODE;
          /* record for unset */
          breaklist[breakcount].name_id = node->op1.name_id;
          breaklist[breakcount].line = breakline;
          breakcount++;
          found = 1;
          break;
        }
      if ( !found )
      { sprintf(msg,"Cannot find instruction on line %d of \"%s\".\n",
           breakline,globals(node->op1.name_id)->name);
        outstring(msg);
      }
      break;
    }

      case UNSET_BREAKPOINT_NODE:
    { 
      struct expnode *proc;
      struct treenode *nodespot;

      if ( node->left ) /* particular */
      {
        int breakline = (int)(*stacktop--);
        /* unmark breakpoint node */
        proc =  &(globals(node->op1.name_id)->value.proc);  
        for ( nodespot = proc->start ; nodespot != proc->root ; nodespot++ )
          if ( nodespot->line_no == breakline )
            nodespot->flags &= ~BREAKPOINT_NODE;
        /* remove from breakpoint list */
        for ( i = 0 ; i < breakcount ; i++ )
          if ( (breaklist[i].name_id == node->op1.name_id) &&
                (breaklist[i].line == breakline) )
          { breaklist[i] = breaklist[--breakcount];
          }
      }
      else /* all */
      {
        for ( i = 0 ; i < breakcount ; i++ )
        { proc =  &(globals(breaklist[i].name_id)->value.proc); 
          for ( nodespot = proc->start ; nodespot != proc->root ; nodespot++ )
            nodespot->flags &= ~BREAKPOINT_NODE;
        }
        breakcount = 0;
      }
      break;
    }

      case DETORUS_NODE: detorus(); recalc(); break;

      case WHEREAMI_COMMAND_NODE: /* for use in debugging or subshells */
        { struct eval_frame *frame = subshell_frame[subshell_depth];
           if ( eval_stack_trace_spot > 0 )
           { 
             for ( i = eval_stack_trace_spot ; i >= 0 ; i-- )
             { struct treenode *node = eval_stack_trace[i];
               if ( node==NULL ) continue;
               sprintf(errmsg,"File %s, %s:%d\n",
                  file_names[node->file_no],frame->base_ex.name,node->line_no);
               outstring(errmsg);
               frame = (struct eval_frame*)(newstack + frame->parent_frame_spot);
             }
             outstring("\n");
          }
          frame = subshell_frame[subshell_depth];
          while ( frame )
          { if ( frame->return_node )
               sprintf(msg,"   %s:%d\n",frame->base_ex.name,frame->return_node->line_no);
            else
               sprintf(msg,"   %s\n",frame->base_ex.name);
            outstring(msg);
            if ( frame->flags & BASE_OF_WHOLE_STACK )
              frame = NULL;
            else
              frame = (struct eval_frame*)(newstack + frame->parent_frame_spot);
          };
          break;
        }

      case FREE_DISCARDS_NODE:
        if ( (node->flags & IN_ELEMENT_LOOP) || 
                 ( this_frame->flags & IN_ELEMENT_LOOP ) )
        { sprintf(errmsg,"free_discards called inside element loop. Ignored.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(1904,errmsg, WARNING);
        }
        else
          free_discards(DISCARDS_ALL); 
        break;

      case SINGLE_LETTER_NODE:
          if ( debugging_flag && node->op1.letter == 'n' )  /* special for debugging prompt */
          { single_step_debugging = 1;
            exit_flag = 1;
            goto the_exit;
            break;
          }
          if ( !(node->flags & IN_ELEMENT_LOOP) &&
               !(this_frame->flags & IN_ELEMENT_LOOP) )
              free_discards(DISCARDS_SOME);
          letter_command(node->op1.letter);
          break;


      case LINEAR_NODE:
          if ( web.modeltype == QUADRATIC )
          { outstring("Changing to LINEAR model. (was QUADRATIC)\n");
            quad_to_linear();
          }
          else if ( web.modeltype == LAGRANGE )
          { outstring("Changing to LINEAR model. (was LAGRANGE)\n");
            lagrange_to_linear();
          }
          break;

      case QUADRATIC_NODE:
          if ( !(node->flags & IN_ELEMENT_LOOP) &&
               !(this_frame->flags & IN_ELEMENT_LOOP) )
              free_discards(DISCARDS_SOME);
          if ( web.modeltype == LINEAR )
          { outstring("Changing to QUADRATIC model. (was LINEAR)\n");
            linear_to_quad(); 
          }
          else if ( web.modeltype == LAGRANGE )
          { outstring("Changing to QUADRATIC model. (was LAGRANGE)\n");
            lagrange_to_quad();
          }
          break;

      case LAGRANGE_NODE:
          if ( !(node->flags & IN_ELEMENT_LOOP) &&
               !(this_frame->flags & IN_ELEMENT_LOOP) )
              free_discards(DISCARDS_SOME);
          if ( web.modeltype == LINEAR )
          { outstring("Changing to LAGRANGE model. (was LINEAR)\n");
            linear_to_lagrange((int)(*stacktop--)); 
          }
          else if ( web.modeltype == QUADRATIC )
          { outstring("Changing to LAGRANGE model. (was QUADRATIC)\n");
            quad_to_lagrange((int)(*stacktop--));
          }
          else
          { sprintf(msg,
                       "Changing to LAGRANGE %d model. (was LAGRANGE %d)\n",
                           (int)stacktop[0],web.lagrange_order);
            outstring(msg);
            lagrange_to_lagrange((int)(*stacktop--));
          }

          break;

      case ELINDEX_NODE: /* id possibly with mpi task number */
         /* creates typeless valid id */
         {
           if ( *stacktop == 0 )
             id = NULLID;
           else 
           { if ( *stacktop > 0.0 )
             id = ((int)(*stacktop)-1);
             else 
             { id = -((int)(*stacktop)+1);         
               invert(id); 
             }
             id |= VALIDMASK;
           }
#ifdef MPI_EVOLVER
         { int task;
           if ( node->right )
              task = (int)(*stacktop--);
           else task = this_task;
           if ( task < 0 || task >= mpi_nprocs )
           { sprintf(errmsg,
                "Illegal task number %d.  Must be between 1 and %d.\n",task,mpi_nprocs);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1915,errmsg,RECOVERABLE);
           }
           id |= (element_id)task << TASK_ID_SHIFT;
         }
#endif
          
           *(element_id *)stacktop = id;
           break;
         }
 
      case PUSH_ELEMENT_ID_NODE:
         *(element_id *)(++stacktop) = node->op1.id;
         break;

      case VALID_ELEMENT_NODE:
         id = get_full_id(node->op1.eltype,*(element_id*)stacktop);
         *stacktop = valid_element(id) ? 1.0 : 0.0; 
         break;
 
      case VALID_CONSTRAINT_NODE:
         { int connum = (int)*stacktop;
           if ( (connum<0) || (connum>=web.maxcon) || !(get_constraint(connum)->attr & IN_USE))
              *stacktop = 0.0;
           else 
              *stacktop = 1.0;
           break;
         }
 
      case VALID_BOUNDARY_NODE:
         { int bnum = (int)*stacktop;
           if ( (bnum<0) || (bnum>=web.bdrymax) || !(web.boundaries[bnum].attr & IN_USE))
              *stacktop = 0.0;
           else 
              *stacktop = 1.0;
           break;
         }

      case LOAD_NODE:
         if ( subshell_depth )
         { sprintf(errmsg,"Can't reload in a subcommand.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(3634,errmsg,WARNING);
           stacktop--;
           break;
         }
         strncpy(loadfilename,*(char**)(stacktop--),sizeof(loadfilename));
#ifdef __cplusplus
         loadstub();  /* indirect throw of exception */
#else
         longjmp(loadjumpbuf,1);
#endif
         break;

      case ADDLOAD_NODE:
        { FILE *newfd;
          int old_read_depth;
          char *name = *(char**)(stacktop--);
          newfd = path_open(name,NOTDATAFILENAME);
          if (newfd == NULL)
          { if ( name[0] )
            {
              sprintf(errmsg,"Cannot open datafile %s.\n",name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(5431,errmsg,RECOVERABLE);
            }
            break; /* continue with old */
          }


          ENTER_GRAPH_MUTEX;

          push_commandfd(newfd,name); /* start #include stack */
          datafile_flag = 1;  /* so parser knows */
          addload_flag = 1;
          datafile_input_flag = 1;  /* so lex input knows */
          cmdptr = 0;
          old_read_depth = read_depth;
          initialize();
          if ( read_depth >= old_read_depth )
            pop_commandfd();
          datafile_flag = 0;
          addload_flag = 0;
          if ( fabs(view[0][0])+fabs(view[1][1])+fabs(view[2][2]) < 1e-25 )
            resize();
  
          LEAVE_GRAPH_MUTEX;
          recalc();
          break;
       }

      case REPLACE_LOAD_NODE:
        { FILE *newfd;
          int old_read_depth;
          char *name = *(char**)(stacktop--);

          newfd = path_open(name,NOTDATAFILENAME);
          if (newfd == NULL)
          { if ( name[0] )
            {
              sprintf(errmsg,"Cannot open datafile %s.\n",name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(5432,errmsg,RECOVERABLE);
            }
            break; /* continue with old */
          }


          ENTER_GRAPH_MUTEX;

          free_element_lists();
#ifdef HHHH
        { body_id b_id;
          facet_id f_id;
          edge_id e_id;
          vertex_id v_id;

          // Dissolve current surface
          FOR_ALL_BODIES(b_id) dissolve_body(b_id);
          FOR_ALL_FACETS(f_id) dissolve_facet(f_id);
          FOR_ALL_EDGES(e_id)  dissolve_edge(e_id);
          FOR_ALL_VERTICES(v_id) dissolve_vertex(v_id);
          free_discards(DISCARDS_ALL);
        }
#endif
          // Now proceed as in addload

          push_commandfd(newfd,name); /* start #include stack */
          datafile_flag = 1;  /* so parser knows */
          addload_flag = 1;
          replace_load_flag = 1;
          datafile_input_flag = 1;  /* so lex input knows */
          cmdptr = 0;
          old_read_depth = read_depth;
          initialize();
          if ( read_depth >= old_read_depth )
            pop_commandfd();
          datafile_flag = 0;
          addload_flag = 0;
          replace_load_flag = 0;
          if ( fabs(view[0][0])+fabs(view[1][1])+fabs(view[2][2]) < 1e-25 )
            resize();
  
          LEAVE_GRAPH_MUTEX;
          recalc();
          break;
       }

      case PERMLOAD_NODE:  /* keep going with same command */
             /* with contortions to preserve current list */
       { struct expnode keeplist;
         struct eval_frame *fr;
         size_t spot;
         if ( subshell_depth )
         { sprintf(errmsg,"Can't reload in a subcommand.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(3635,errmsg,WARNING);
           stacktop--;
           break;

         }

         /* make sure all parent frames permanent (except very first frame
            corresponding to original user command */
         for ( fr = this_frame; fr != NULL ; 
            fr = (struct eval_frame*)(newstack + fr->parent_frame_spot) )
         { if ( !(fr->flags & BASE_OF_WHOLE_STACK) && !(fr->base_ex.start->flags & PERMNODE) )
           { strcpy(errmsg, "Calling permload in non-permanent command.\n");
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(3364,errmsg, RECOVERABLE);
           }
           if ( fr->flags & BASE_OF_WHOLE_STACK ) break; 
         }
         if ( !(fr->base_ex.start->flags & PERMNODE) )
         { /* have to kludge to preserve current command list */
           memset(&keeplist,0,sizeof(keeplist));
           perm_tree_copy(&keeplist,fr->base_ex.root);
           spot = (*(fr->basenode) - (fr->base_ex.start));
           *(fr->basenode) = keeplist.start + spot; 
           memset(&(fr->base_ex),0,sizeof(struct expnode)); /* so doesn't do free_expr()later */        
           fr->base_ex = keeplist;
         }
         strncpy(loadfilename,*(char**)(stacktop--),sizeof(loadfilename));
         startup(loadfilename);
         exec_commands(commandfd,"Enter command: "); /* from end of datafile */
         /* keeplist permanently allocated, but we'll live with memory leak
            as price for not doing shenanigans every eval() just to avoid leak. */
       }
       break;

      case FUNCTION_CALL_NODE:
        { struct global *g = globals(node->op1.name_id);

          if ( g->value.proc.root == NULL )
          { sprintf(errmsg,
            "Function \"%s\" definition has not been executed yet.\n",g->name);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2621,errmsg,RECOVERABLE);
          }
          
          ex_current = g->value.proc;
          
          PROF_EVAL_PUSH_AND_SWITCH(&g->value.proc);
          PUSH_TRACE;
          return_node = node;
          node = g->value.proc.start;
          break;
        }
        
      case FUNCTION_CALL_RETURN_NODE:
        { PROF_EVAL_POP;
          stacktop -= node->op2.argcount; /* pop arguments */
          *(++stacktop) = return_value;
          POP_TRACE;
          break;
        }

      case PROCEDURE_CALL_NODE: /* with arguments */
        { struct global *g = globals(node->op1.name_id);
          if ( g->value.proc.root == NULL )
          { sprintf(errmsg,
            "Procedure \"%s\" definition has not been executed yet.\n",g->name);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2622,errmsg,RECOVERABLE);
          }

          ex_current = g->value.proc;
                    
          PROF_EVAL_PUSH_AND_SWITCH(&g->value.proc);
          PUSH_TRACE;
          return_node = node;
          node = g->value.proc.start;             
          break;
        }
        
      case PROCEDURE_CALL_RETURN_NODE:    
          PROF_EVAL_POP;
          stacktop -= node->op2.argcount; /* pop arguments */
          POP_TRACE;
          break;
        

      case PROCEDURE_NODE:
        { struct global *g = globals(node->op1.name_id);
          PROF_EVAL_END(ex_profiling);
          PUSH_TRACE;
          eval(&(g->value.proc),NULL,NULLID,NULL);
          PROF_EVAL_START(ex_profiling);
          POP_TRACE;
          break;
        }

      case PERM_PROCEDURE_NODE:
        { struct expnode *newex;
          struct global *g = perm_globals(node->op1.name_id);         
          PROF_EVAL_END(ex_profiling);
          newex = &(g->value.proc);
          newex->start->flags |= PERMNODE;
          PUSH_TRACE;
          eval(newex,NULL,NULLID,NULL);
          POP_TRACE;
          PROF_EVAL_START(ex_profiling);
          break;
        }

      case CMDLIST_NODE: case COMMAND_BLOCK_NODE:
          /* no action, just holds tree together */
          break;
   
      case DECLARE_LOCAL_NODE: break;
      case LOCAL_LIST_START_NODE: break;
    
      case SINGLE_REDEFD_NODE:
          if ( single_redefine[node->op1.letter].start )
          {
            PROF_EVAL_END(ex_profiling);
            PUSH_TRACE;
            eval(&single_redefine[node->op1.letter],NULL,NULLID,NULL);
            POP_TRACE;
            PROF_EVAL_START(ex_profiling);
          }
          else letter_command(node->op1.letter);
          break;


      case MATRIX_INVERSE_NODE: 
        { REAL *datastart2 = *(REAL**)(stacktop--);
          REAL *datastart1 = *(REAL**)(stacktop--);
          struct array *a = get_name_arrayptr(node->op1.name_id,newstack+localstack,localbase);
          struct array *b = get_name_arrayptr(node->op2.name_id,newstack+localstack,localbase);
 
           *(++stacktop) = (REAL)matrix_inverse_command(a,
                 b,datastart1,datastart2);
        }
           break;

      case MATRIX_MULTIPLY_NODE: 
        { REAL *datastart3 = *(REAL**)(stacktop--);
          REAL *datastart2 = *(REAL**)(stacktop--);  
          REAL *datastart1 = *(REAL**)(stacktop--); 
          struct array *a = get_name_arrayptr(node->op1.name_id,newstack+localstack,localbase);
          struct array *b = get_name_arrayptr(node->op2.name_id,newstack+localstack,localbase);
          struct array *c = get_name_arrayptr(node->op3.name_id,newstack+localstack,localbase);
          matrix_multiply_command(a,b,c,datastart1,datastart2,datastart3);
        }
        break;

      case MATRIX_DETERMINANT_NODE:
        { REAL *datastart1 = *(REAL**)(stacktop--);
          struct array *a = get_name_arrayptr(node->op1.name_id,newstack+localstack,localbase);
          *(++stacktop) = matrix_determinant_command(a,datastart1);
        }
        break;

      case BACKQUOTE_START_NODE:
        { struct expnode bqnode = ex_current;
          bqnode.start = node;
          bqnode.root = node+node->op1.skipsize;
          bqnode.locals = localbase;

          PROF_EVAL_END(ex_profiling);
          PUSH_TRACE;
          eval(&bqnode,params,self_id,this_frame);
          POP_TRACE;
          PROF_EVAL_START(ex_profiling);
          node += node->op1.skipsize; /* skip what was evaluated */
        }
        break;
      case BACKQUOTE_END_NODE: break;  /* just a placeholder */
      case ACOMMANDEXPR_NODE: /* backquoted command at start of expression */
           break;

      case INDEXSET_NODE: break; /* just accumulate index values */
      case DIMENSIONSET_NODE: break; /* just accumulate index values */
      case DEFINE_FIXED_LOCAL_ARRAY_NODE: break; /* was allocated on stack */

      case SHOW_NODE:
      case SHOW_EXPR_NODE:
      { int etype; /* element type */  
        ENTER_GRAPH_MUTEX
        where = node + node->op1.skipsize;
        if ( where->type == WHERE_NODE ) /* condition */
          { /* copy over expression */
             enode = where + where->left; /* NEXT */
             eroot = where + where->right;
             etype = enode[enode->left].op1.eltype;
             show_expr[etype] = show_expr_table + etype;
             tree_copy(show_expr[etype],eroot);
             sprintf(show_expr[etype]->name,"show expression for %s.",
                typenames[etype]);
             /* can use first slot to record type of element */
             /* element location */
             show_expr[etype]->start->op2.eltype = enode->op2.eltype;
             enode += enode->left; /* INIT */
             show_expr[etype]->start->op1.eltype = enode->op1.eltype;
             // copy over local variable data
             show_expr[etype]->locals = (struct locallist_t*)mycalloc(1,sizeof(struct locallist_t));
             *(show_expr[etype]->locals) = *(ex_current.locals);
             show_expr[etype]->locals->list = 
                  (struct localvar_t *)mycalloc(ex_current.locals->count,
                             sizeof(struct localvar_t));
             memcpy(show_expr[etype]->locals->list,ex_current.locals->list,ex_current.locals->count*
                              sizeof(struct localvar_t));
          }
         else
            { etype =  where[where->left].op1.eltype;
              tree_copy(show_expr[etype],NULL);
              show_expr[etype] = NULL;
            }
         /* save for dump */ 
         tree_copy(show_command+etype,node+node->op1.skipsize+1);
         LEAVE_GRAPH_MUTEX
         if ( (node->type == SHOW_NODE) && !OOGL_flag && !go_display_flag)
            do_show();
         else update_display();
         node += node->op1.skipsize; /* skip over expression */
         break;
       }

      case UNREDEFINE_SINGLE_NODE:
        free_expr(&single_redefine[node->op1.letter]);
        break;
          
      case REDEFINE_SINGLE_NODE:
        tree_copy(&single_redefine[node->op1.letter],node+node->op2.jumpsize);
        sprintf(single_redefine[node->op1.letter].name,"redefined command '%c'",
             node->op1.letter);
        single_redefine[node->op1.letter].flag = USERCOPY;
        locals_copy(&(single_redefine[node->op1.letter].locals),
                node->op5.locals); 
        node += node->op2.jumpsize; /* skip over procedure */
        break;

      case SET_PROCEDURE_NODE:
        g = globals(node->op1.name_id);
        free_expr(&g->value.proc);
        tree_copy(&g->value.proc, node+node->op2.jumpsize);
        strcpy(g->value.proc.name, g->name);
        g->attr.procstuff.proc_timestamp = proc_timestamp++;
        locals_copy(&(g->value.proc.locals),node->op5.locals);
        node += node->op2.jumpsize; /* skip over procedure */
        break;

      case SET_PERM_PROCEDURE_NODE:
        g = perm_globals(node->op1.name_id);
        perm_free_expr(&g->value.proc);
        perm_tree_copy(&g->value.proc, node+node->op2.jumpsize);
        strcpy(g->value.proc.name, g->name);
        g->attr.procstuff.proc_timestamp = proc_timestamp++;
        locals_copy_perm(&(g->value.proc.locals),node->op5.locals);
        node += node->op2.jumpsize; /* skip over procedure */
        break;

      case SET_FUNCTION_NODE: break;
      case FUNCTION_HEAD_NODE: break;
      case ARGLIST_NODE: break;

      case FUNCTION_DEF_START_NODE:
        g = globals(node->op1.name_id);
        free_expr(&g->value.proc);
        tree_copy(&g->value.proc, node+node->op2.jumpsize);
        g->value.proc.start[2].type = FUNCTION_START_NODE;
        strcpy(g->value.proc.name, g->name);
        g->attr.procstuff.proc_timestamp = proc_timestamp++;
        locals_copy(&(g->value.proc.locals),node->op5.locals);
        node += node->op2.jumpsize; /* skip over procedure */
        break;

      case FUNCTION_PROTO_START_NODE:
        node += node->op2.jumpsize; /* skip over stuff */
        break;

      case FUNCTION_START_NODE:  /* function entry code */
        /* copy arguments over to local variable space */
          memcpy((char*)(stacktop-localcount+1),
           (char*)(((REAL*)this_frame)-(node->op3.argcount+PROF_SPACE)),
             node->op3.argcount*sizeof(REAL));
        break;

      case SET_ARGSPROC_NODE: break;
      case PROCEDURE_HEAD_NODE: break;

      case PROCEDURE_DEF_START_NODE:
        g = globals(node->op1.name_id);
        free_expr(&g->value.proc);
        tree_copy(&g->value.proc, node+node->op2.jumpsize);
        g->value.proc.start[2].type = PROCEDURE_START_NODE;
        strcpy(g->value.proc.name, g->name);
        g->attr.procstuff.proc_timestamp = proc_timestamp++;
        locals_copy(&(g->value.proc.locals),node->op5.locals);
        node += node->op2.jumpsize; /* skip over procedure */
        break;

      case PROCEDURE_PROTO_START_NODE:
        node += node->op2.jumpsize; /* skip over stuff */ 
        break;

      case PROCEDURE_START_NODE:  /* function entry code */
        /* copy arguments over to local variable space */
        if ( node->op3.argcount )
          memcpy((char*)(stacktop-localcount+1),
           (char*)(((REAL*)this_frame)-(node->op3.argcount+PROF_SPACE)),
             node->op3.argcount*sizeof(REAL));
        break;

      case RETURN_NODE:
        if ( node->left ) 
        { newstack[localstack + localbase->totalsize] = *stacktop;
          stacktop = newstack + localstack + localbase->totalsize;
        }
        else 
          stacktop = newstack + localstack + localbase->totalsize - 1;
        node = ex_current.root; /* since next node should be FINISHED_NODE */
        if ( node[1].type != FINISHED_NODE )
        { sprintf(errmsg,"Internal error: no FINISH node after RETURN\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(4578,errmsg, RECOVERABLE);
        }
        break;
        

         /********************/
         /* repeated command */
         /********************/

      case REPEAT_INIT_NODE:
        /* target on stack, then iteration count */
        gocount = (int)*stacktop; 
        *(++stacktop) = 0; /* iterations done */
        if ( gocount <= 0 ) 
        { stacktop -= 2; /* pop counts */
          node += node->op1.skipsize; /* skip loop */
          gocount = 1;
        }
        break;

      case REPEAT_NODE:
      { struct treenode *rnode = node + node->left;
       
        stacktop[0] += 1;
        gocount = (int)(stacktop[-1] - stacktop[0]);
#if defined(MAC_APP) || defined(MAC_CW)
        break_check();
#endif
        if ( breakflag  ) 
        { stacktop -= 2; breakflag = 0; break; }
        if ( (gocount > 0) && !breakflag )
          { node = rnode;  /* do again */
             /* loop increment will step over REPEAT_INIT_ node */
          }
        else 
        { stacktop -= 2; /* pop gocount and total count */
          gocount = 1;   /* so prints as 1 when no count given */
        }
      }
      break;


      /*******************/
      /* flow of control */
      /*******************/

      case IFTEST_NODE:
      case COND_TEST_NODE:
        if ( *(stacktop--) == 0. )
        { /* jump */
          node += node->op1.skipsize;
        }
        break;

      case IF_NODE:
      case COND_EXPR_NODE:
         /* did first command, so skip second */
         node += node->op1.skipsize;
         break;

      case CONTINUE_NODE:
         node += node->op1.skipsize; /* back to loop top */
         stacktop = newstack + localstack + *(size_t*)(newstack + localstack+node->stackpos);
         node += node->op4.contjump; /* jump to expr */
         break;

      case BREAK_NODE:
         node += node->op1.skipsize; /* back to loop top */
         stacktop = newstack + localstack + *(size_t*)(newstack + localstack+node->stackpos);
         node += node->op3.breakjump; /* jump to end */
         break;

      case WHILE_TOP_NODE:
         /* break and continue jumps */
         *(size_t*)(newstack + localstack + node->stackpos) = stacktop - (newstack + localstack) - 1;
         /* jumptest if expr false */
         if ( (*(stacktop--) == 0.) || breakflag )
           node += node->op1.skipsize;
         break;

      case WHILE_END_NODE:
         /* loop back */
         node += node->op1.skipsize;
         break;

      case DO_TOP_NODE: 
         /* break and continue jumps */
         *(size_t*)(newstack + localstack + node->stackpos) = stacktop - (newstack + localstack);
         break;

      case DO_ENTRY_NODE: 
         /* break and continue jumps */
         *(size_t*)(newstack + localstack + node->stackpos) = stacktop - (newstack + localstack);
         break;

      case DO_END_NODE:
         /* loop back if true */
         if ( *(stacktop--) && !breakflag )
           node += node->op1.skipsize;
         break;

      case FOR_ENTRY_NODE:
         /* break and continue jumps */
         *(size_t*)(newstack + localstack + node->stackpos) = stacktop - (newstack + localstack);
         break;

      case FOR_HEAD_NODE: 
         if ( *(stacktop--) && !breakflag )
           node += node->op1.skipsize;  /* loop body */
         else
           node += node->op2.jumpsize; /* break out of loop */
         break;

      case FOR_TOP_NODE:
         node += node->op1.skipsize; /* jump to test expr */
         break;

      case FOR_END_NODE:
         node += node->op1.skipsize; /* jump to increment command */
         break;


      
         /*******************/
         /* aggregate verbs */
         /*******************/

      case LIST_NODE:
        oldquiet = quiet_flag; quiet_flag = 0;
        switch ( node->op2.eltype )
        { 
           case VERTEX:
             vertex_dump(positive_id(q_id),outfd);
             break;

           case EDGE:
             edge_dump(positive_id(q_id),outfd);
             break;

           case FACET:
             facet_dump(positive_id(q_id),outfd);
             break;

           case BODY:
             body_dump(positive_id(q_id),outfd);
             break;

           case FACETEDGE:
             facetedge_dump(positive_id(q_id),outfd);
             break;

           default: 
             sprintf(errmsg,"Bad LIST element type in %s.\n",ex_current.name);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             quiet_flag = oldquiet;   
             kb_error(1255,errmsg,RECOVERABLE);

         }
        quiet_flag = oldquiet;   
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      /***********************
      * commands with counts *
      ***********************/
      case REFINE_NODE:
         switch ( node->op2.eltype )
         { 
            case EDGE:
              n = valid_id(edge_refine(q_id));
              if ( web.counts_reported & edge_refine_count_bit )
              { web.edge_refine_count = 0; 
                web.counts_reported &= ~edge_refine_count_bit;
              }
              if ( n )
              { web.edge_refine_count += n; 
                web.counts_changed |= edge_refine_count_bit;
                recalc_flag = 1;
              }
              break;
            case FACET:
              face_triangulate(q_id,FACET_EDGES);
              n = 1;
              if ( web.counts_reported & facet_refine_count_bit )
              { web.facet_refine_count = 0; 
                web.counts_reported &= ~facet_refine_count_bit;
              }
              if ( n )
              { web.facet_refine_count += n; 
                web.counts_changed |= facet_refine_count_bit;
                recalc_flag = 1;
              }
              break;
            default: 
              sprintf(errmsg,"Bad refine element type in %s.\n",ex_current.name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(1256,errmsg, RECOVERABLE);

         }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case POP_NODE:
         n = 0;
         if ( id_type(q_id) == EDGE )
         { n = pop_one_edge(q_id); 
           if ( web.counts_reported & edge_pop_count_bit )
           { web.edge_pop_count = 0; 
             web.counts_reported &= ~edge_pop_count_bit;
           }
           if ( n )
           { web.edge_pop_count += n; 
             web.counts_changed |= edge_pop_count_bit;
           }
         }
         else if ( id_type(q_id) == VERTEX )
         { n = pop_given_vertex(q_id); 
           if ( web.counts_reported & vertex_pop_count_bit )
           { web.vertex_pop_count = 0; 
             web.counts_reported &= ~vertex_pop_count_bit;
           }
           if ( n )
           { web.vertex_pop_count += n; 
             web.counts_changed |= vertex_pop_count_bit;
           }
         }
         else 
         { sprintf(errmsg,"Only vertices and edges poppable.\n");
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(3356,errmsg, RECOVERABLE);
         }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         if ( n )  recalc_flag = 1; 
         break;

      case POP_TRI_TO_EDGE_NODE:
         n = pop_tri_to_edge(q_id); 
         if ( web.counts_reported & pop_tri_to_edge_count_bit )
         { web.pop_tri_to_edge_count = 0;
           web.counts_reported &= ~pop_tri_to_edge_count_bit;
         }
         if ( n )
         { web.pop_tri_to_edge_count += n; 
           web.counts_changed |= pop_tri_to_edge_count_bit;
         }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         if ( n )  recalc_flag = 1; 
         break;

      case POP_EDGE_TO_TRI_NODE:
         n = pop_edge_to_tri(q_id); 
         if ( web.counts_reported & pop_edge_to_tri_count_bit )
           { web.pop_edge_to_tri_count = 0; 
             web.counts_reported &= ~pop_edge_to_tri_count_bit;
           }
         if ( n )
           { web.pop_edge_to_tri_count += n; 
             web.counts_changed |= pop_edge_to_tri_count_bit;
           }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         if ( n )  recalc_flag = 1; 
         break;

      case POP_QUAD_TO_QUAD_NODE:
         n = pop_quad_to_quad(q_id); 
         if ( web.counts_reported & pop_quad_to_quad_count_bit )
           { web.pop_quad_to_quad_count = 0; 
             web.counts_reported &= ~pop_quad_to_quad_count_bit;
           }
         if ( n )
           { web.pop_quad_to_quad_count += n; 
             web.counts_changed |= pop_quad_to_quad_count_bit;
           }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         if ( n )  recalc_flag = 1;
         break;

      case EDGESWAP_NODE:
         n = edgeswap(q_id); 
         if ( web.counts_reported & edgeswap_count_bit )
           { web.edgeswap_count = 0; 
             web.counts_reported &= ~edgeswap_count_bit;
           }
         if ( n )
           { web.edgeswap_count += n; 
             web.counts_changed |= edgeswap_count_bit;
           }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         if ( n ) recalc_flag = 1;
         break;

      case T1_EDGESWAP_NODE:
         n = t1_edgeswap(q_id); 
         if ( web.counts_reported & t1_edgeswap_count_bit )
           { web.t1_edgeswap_count = 0; 
             web.counts_reported &= ~t1_edgeswap_count_bit;
           }
         if ( n )
           { web.t1_edgeswap_count += n; 
             web.counts_changed |= t1_edgeswap_count_bit;
           }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         if ( n )  recalc_flag = 1;
         break;

      case EQUIANGULATE_NODE:
         n = equiangulate_edge(q_id); 
         if ( web.counts_reported & equi_count_bit )
           { web.equi_count = 0; 
             web.counts_reported &= ~equi_count_bit;
           }
         if ( n )
           { web.equi_count += n; 
             web.counts_changed |= equi_count_bit;
           }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         if ( n ) recalc_flag = 1; 
         
         break;

      case DISSOLVE_NODE:
         switch ( node->op2.eltype )
         {  case VERTEX: 
              n = dissolve_vertex(q_id); 
              if ( web.counts_reported & vertex_dissolve_count_bit )
              { web.vertex_dissolve_count = 0; 
                web.counts_reported &= ~vertex_dissolve_count_bit;
              }
              if ( n )
              { web.vertex_dissolve_count += n; 
                web.counts_changed |= vertex_dissolve_count_bit;
                recalc_flag = 1;
              }
              break;
            case EDGE:
              n = dissolve_edge(q_id); 
              if ( web.counts_reported & edge_dissolve_count_bit )
              { web.edge_dissolve_count = 0; 
                web.counts_reported &= ~edge_dissolve_count_bit;
              }
              if ( n )
              { web.edge_dissolve_count += n; 
                web.counts_changed |= edge_dissolve_count_bit;
                recalc_flag = 1;
              }
              break;
            case FACET:
              n = dissolve_facet(q_id); 
              if ( web.counts_reported & facet_dissolve_count_bit )
              { web.facet_dissolve_count = 0; 
                web.counts_reported &= ~facet_dissolve_count_bit;
              }
              if ( n )
              { web.facet_dissolve_count += n; 
                web.counts_changed |= facet_dissolve_count_bit;
                recalc_flag = 1;
              }
              break;
            case BODY:
              n = dissolve_body(q_id); 
              if ( web.counts_reported & body_dissolve_count_bit )
              { web.body_dissolve_count = 0; 
                web.counts_reported &= ~body_dissolve_count_bit;
              }
              if ( n )
              { web.body_dissolve_count += n; 
                web.counts_changed |= body_dissolve_count_bit;
                recalc_flag = 1;
              }
              break;
          }
          node += node->op1.skipsize - 1;  /* back to start of loop */
          break;


      case REVERSE_ORIENTATION_NODE:
         switch ( node->op2.eltype )
         { 
            case EDGE:
              reverse_orientation_edge(q_id); 
              if ( web.counts_reported & edge_reverse_count_bit )
              { web.edge_reverse_count = 0; 
                web.counts_reported &= ~edge_reverse_count_bit;
              }
              
              web.edge_reverse_count += 1; 
              web.counts_changed |= edge_reverse_count_bit;
              recalc_flag = 1;
              
              break;
            case FACET:
              reverse_orientation_facet(q_id); 
              if ( web.counts_reported & facet_reverse_count_bit )
              { web.facet_reverse_count = 0; 
                web.counts_reported &= ~facet_reverse_count_bit;
              }
              
              web.facet_reverse_count += 1; 
              web.counts_changed |= facet_reverse_count_bit;
              recalc_flag = 1;
              
              break;
          }
          node += node->op1.skipsize - 1;  /* back to start of loop */
          break;

      case FIX_NODE: 
         if ( web.counts_reported & fix_count_bit )
         { web.fix_count = 0; 
           web.counts_reported &= ~fix_count_bit;
         }
         if ( !(get_attr(q_id)&FIXED) )
         { set_attr(q_id,FIXED); 
           web.fix_count += 1; 
           web.counts_changed |= fix_count_bit;
         }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case UNFIX_NODE:
         if ( web.counts_reported & unfix_count_bit )
         { web.unfix_count = 0; 
           web.counts_reported &= ~unfix_count_bit;
         }
         if ( get_attr(q_id) & FIXED )
         { unset_attr(q_id,FIXED); 
           web.unfix_count += 1; 
           web.counts_changed |= unfix_count_bit;
         }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case VERTEX_AVERAGE_NODE:
         if ( new_vertex_average(q_id,VOLKEEP) ) recalc_flag = 1;
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case RAW_VERTEX_AVERAGE_NODE:
         if ( new_vertex_average(q_id,NOVOLKEEP) ) recalc_flag = 1;
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case RAWEST_VERTEX_AVERAGE_NODE:
         if ( new_vertex_average(q_id,RAWEST) ) recalc_flag = 1;
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case DELETE_NODE:
         id = q_id;
         switch ( node->op2.eltype )
         { 
            case EDGE:
              n = delete_edge(id);
              if ( n ) 
              { if ( !valid_element(id) )
                { sprintf(errmsg,"Really strange configuration around deleted edge %s\n",
                     ELNAME(id));
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                                file_names[node->file_no],node->line_no);
                  kb_error(1919,errmsg,WARNING);
                }
                else
                  free_element(id);
              }
              if ( web.counts_reported & edge_delete_count_bit )
              { web.edge_delete_count = 0; 
                web.counts_reported &= ~edge_delete_count_bit;
              }
              if ( n )
              { web.edge_delete_count += n; 
                web.counts_changed |= edge_delete_count_bit;
                recalc_flag = 1;
              }
              break;
            case FACET:
              n = delete_facet(id);
              if ( web.counts_reported & facet_delete_count_bit )
              { web.facet_delete_count = 0; 
                web.counts_reported &= ~facet_delete_count_bit;
              }
              if ( n > 0 )
              { web.facet_delete_count += n; 
                web.counts_changed |= facet_delete_count_bit;
                recalc_flag = 1;
              }
              if ( n < 0 ) /* from string_delete_edge, partial elim */
              { web.edge_delete_count += -n;
                web.counts_reported &= ~edge_delete_count_bit;
                recalc_flag = 1;
              }
                
              break;

            default: 
              sprintf(errmsg,"Bad delete element type in %s.\n",ex_current.name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(1257,errmsg, RECOVERABLE);
         }
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

         /***************/
         /* expressions */
         /***************/

      case REPLACECONST_NODE:
         *stacktop = node->op1.real;
        break;

      case PUSHCONST_NODE:
         *++stacktop = node->op1.real;
        break;

      case PUSHDELTA_NODE:
        { struct global *g = globals(node->op1.name_id);
          *++stacktop = g->attr.varstuff.delta;
          break;
        }

      case PUSH_PARAM_SCALE_NODE:
        { struct global *g = globals(node->op1.name_id);
          *++stacktop = g->attr.varstuff.pscale;
          break;
       }
      case PUSH_PARAM_FIXED_NODE:
        { struct global *g = globals(node->op1.name_id);
          *++stacktop = (g->flags & OPTIMIZING_PARAMETER)?0.0:1.0;
          break;
        }
      case PUSH_PARAM_FORCE_NODE:
      { int i;
          for ( i = 0 ; i < optparamcount ; i++ )
            if ( optparam[i].pnum == node->op1.name_id )
            {  *++stacktop = optparam[i].grad;
               break;
            }
          if ( i == optparamcount ) *++stacktop = 0.0;
          break;
       }
      case PUSH_PARAM_VELOCITY_NODE:
      { int i;
          for ( i = 0 ; i < optparamcount ; i++ )
            if ( optparam[i].pnum == node->op1.name_id )
            {  *++stacktop = optparam[i].velocity;
               break;
            }
          if ( i == optparamcount ) *++stacktop = 0.0;
          break;
       }
      case PUSH_PARAM_EXTRA_NODE:
        { int i;
          for ( i = 0 ; i < optparamcount ; i++ )
            if ( optparam[i].pnum == node->op1.name_id )
            { switch ( node->op2.extranum )
              { case V_VELOCITY_ATTR:
                  *++stacktop = optparam[i].velocity;
                  break;
                case V_FORCE_ATTR:
                  *++stacktop = optparam[i].grad;
                  break;
              }
              break;
            }
          if ( i == optparamcount ) *++stacktop = 0.0;
          break;
       }

      case PUSHGLOBAL_NODE:
      case STRINGGLOBAL_NODE:
      case PUSH_PERM_GLOBAL_NODE:
      case PERM_STRINGGLOBAL_NODE:
        { struct global *g = globals(node->op1.name_id);
          if ( g->flags & GLOB_LOCALVAR )
             *++stacktop = newstack[localstack+g->value.offset];
          else if ( g->flags & FILE_VALUES )
             *++stacktop = g->value.file.values[int_val];
          else if ( g->flags & STRINGVAL )
          { int pp = (sizeof(REAL)+sizeof(char*)-1)/sizeof(char*);
            int nn; 
            stacktop++;
            for ( nn = 0 ; nn < pp ; nn++ )
               ((char **)stacktop)[nn] = g->value.string;
          }
          else if ( g->flags & INTERNAL_NAME )
          { if ( g->flags & INTVAL )
              *++stacktop = *(int*)(g->value.dataptr);
            else if ( g->flags & REALVAL )
              *++stacktop = *(REAL*)(g->value.dataptr);
            else
            { sprintf(errmsg,
               "Internal error: Internal variable %s type not set.\n",g->name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(2851,errmsg,RECOVERABLE);
            }
          }
          else
            *++stacktop = g->value.real;
        }
        break;

      case PUSHPI_NODE:
         *++stacktop = M_PI;
        break;

      case PUSHE_NODE:
         *++stacktop = M_E;
        break;

      case PUSHG_NODE:
         *++stacktop = web.gravflag ? web.grav_const : 0.0;
        break;

      case PUSHPARAM_NODE:
         *++stacktop = params[node->op1.coordnum];
        break;

      case USERFUNC_NODE:
        *++stacktop = (*userfunc[node->op1.userfunc])(params);
        break;

      case DYNAMIC_LOAD_FUNC_NODE:
        if ( ! params )
        { sprintf(errmsg,
           "Must use dynamic load function %s in context with parameters.\n",
             globals(node->op2.name_id)->name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2059,errmsg,RECOVERABLE);
        } else
        (*node->op1.funcptr)(FUNC_VALUE,params,(struct dstack*)(++stacktop));
        break;

      case GT_NODE:
        stacktop--;
        stacktop[0] = (REAL)(stacktop[0] > stacktop[1]);
        break;

      case LT_NODE:
        stacktop--;
        stacktop[0] = (REAL)(stacktop[0] < stacktop[1]);
        break;

      case LE_NODE:
        stacktop--;
        stacktop[0] = (REAL)(stacktop[0] <= stacktop[1]);
        break;

      case GE_NODE:
        stacktop--;
        stacktop[0] = (REAL)(stacktop[0] >= stacktop[1]);
        break;

      case NE_NODE:
        stacktop--;
        stacktop[0] = (REAL)(stacktop[0] != stacktop[1]);
        break;

      case EQ_NODE:
        stacktop--;
        stacktop[0] = (REAL)(stacktop[0] == stacktop[1]);
        break;

      case AND_NODE: /* short-circuit */
        if ( *stacktop == 0.0 )
        { *(++stacktop) = 0.0;
          node += node->op1.skipsize; /* leave 0 as result */
        }
        break;

      case CONJUNCTION_END_NODE:    
        /* short-circuiting results in second arg being answer */
        /* get proper 1 for true */
        stacktop--;
        *stacktop = stacktop[1];
        if ( *stacktop ) *stacktop = 1.0;
        break;

      case OR_NODE: /* short-circuit */
        if ( *stacktop != 0.0 )
        { *(++stacktop) = 1.0;
          node += node->op1.skipsize; /* leave as result */
        }
        break;

      case NOT_NODE:
        stacktop[0] = (REAL)(!stacktop[0]);
        break;

      case PLUS_NODE:
        stacktop--;
        stacktop[0] += stacktop[1];
        break;

      case MINUS_NODE:
      case EQUATE_NODE:
        stacktop--;
        stacktop[0] -= stacktop[1];
        break;

      case TIMES_NODE:
        stacktop--;
        stacktop[0] *= stacktop[1];
        break;

      case DIVIDE_NODE:
        stacktop--;
        if ( stacktop[1] == 0.0 ) 
        { if ( valid_id(self_id) ) 
          sprintf(errmsg,"Division by zero in %s, %s %s.\n",ex_current.name, 
             typenames[id_type(self_id)],ELNAME(self_id));
          else sprintf(errmsg,"Division by zero in %s.\n",ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(1258,errmsg,RECOVERABLE);
        }
        stacktop[0] /= stacktop[1];
        break;

      case REALMOD_NODE:
        stacktop--;
        if ( stacktop[1] == 0.0 ) 
        { if ( valid_id(self_id) )
            sprintf(errmsg,"Modulus base zero in %s, %s %s.\n",ex_current.name,
             typenames[id_type(self_id)],ELNAME(self_id));
          else sprintf(errmsg,"Modulus base zero in %s.\n",ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(1259,errmsg,RECOVERABLE);
        }
        stacktop[0] = stacktop[0] - floor(stacktop[0]/stacktop[1])
                              *stacktop[1];
        break;

      case IMOD_NODE:
        stacktop--;
        if ( stacktop[1] == 0.0 ) 
        { if ( valid_id(self_id) )
            sprintf(errmsg,"Modulus base zero in %s, %s %s.\n",ex_current.name,
             typenames[id_type(self_id)],ELNAME(self_id));
          else sprintf(errmsg,"Modulus base zero in %s.\n",ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(1260,errmsg,RECOVERABLE);
        }
        stacktop[0] = floor(stacktop[0]) - 
            floor(floor(stacktop[0])/floor(stacktop[1]))
                              *floor(stacktop[1]);
        break;

      case IDIV_NODE:
        stacktop--;
        if ( (int)stacktop[1] == 0 ) 
        { if ( valid_id(self_id) )
            sprintf(errmsg,"Division by zero in %s, %s %s.\n",ex_current.name,
             typenames[id_type(self_id)],ELNAME(self_id));
          else sprintf(errmsg,"Division by zero in %s.\n",ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(1261,errmsg,RECOVERABLE);
        }
        stacktop[0] = (REAL)((int)(stacktop[0])/(int)(stacktop[1]));
        break;

      case INTPOW_NODE:
         /* cases n = 0,1,2 taken care of in parsing */
         x = *stacktop;
         k = node->op1.intpow < 0 ? -node->op1.intpow : node->op1.intpow;
         for ( n = 1 ; n < k ; n++ )
           *stacktop *= x;
         if ( node->op1.intpow < 0 )
         { if ( *stacktop == 0.0 ) 
           { if ( valid_id(self_id) )
               sprintf(errmsg,"Negative power (%d) of zero in %s, %s %s.\n",
                 node->op1.intpow,ex_current.name,
                 typenames[id_type(self_id)],ELNAME(self_id));
             else sprintf(errmsg,"Negative power (%d) of zero in %s.\n",
                 node->op1.intpow,ex_current.name);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1262,errmsg,RECOVERABLE);
           }
           else
             *stacktop = 1/(*stacktop);
         } 
         break;

      case POW_NODE:
        stacktop--;
        if ( (stacktop[0] < 0.0) && (floor(stacktop[1]) != stacktop[1]) )
        { if ( valid_id(self_id) )
            sprintf(errmsg,
              "Non-integer power of a negative number in %s, %s %s.\n",
              ex_current.name, typenames[id_type(self_id)],ELNAME(self_id));
          else sprintf(errmsg,"Non-integer power of a negative number in %s.\n",
              ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2060,errmsg,RECOVERABLE);
        }
        if ( stacktop[0] == 0.0 )
        { if ( stacktop[1] >= 0.0 ) *stacktop = 0.0;
          else *stacktop = 1e30;
        }
        else  *stacktop = pow(stacktop[0],stacktop[1]);
        /* Note: pow recognizes integer powers, so OK for neg x */
        break;

      case MAXIMUM_NODE:
        stacktop--;
         *stacktop = (stacktop[0] > stacktop[1]) ? stacktop[0] : stacktop[1];
        break;

      case MINIMUM_NODE:
        stacktop--;
         *stacktop = (stacktop[0] < stacktop[1]) ? stacktop[0] : stacktop[1];
        break;

      case WRAP_COMPOSE_NODE:
          stacktop--;
          if ( sym_compose )
          *stacktop = (REAL)(*sym_compose)((unsigned int)stacktop[0],(unsigned int)stacktop[1]);
          break;

      case WRAP_INVERSE_NODE:
          if ( sym_compose )
          *stacktop = (REAL)(*sym_inverse)((unsigned int)stacktop[0]);
           break;

      case ATAN2_NODE:
        stacktop--;
         *stacktop = atan2(stacktop[0],stacktop[1]);
        break;

      case SQR_NODE:
         *stacktop *= *stacktop;
        break;

      case SQRT_NODE:
        if ( *stacktop < 0.0 )
        { if ( *stacktop > -100*machine_eps ) *stacktop = 0.0;
          else 
          { if ( valid_id(self_id) )
             sprintf(errmsg,"Square root of negative number %g in %s, %s %s.\n",*stacktop,
               ex_current.name,typenames[id_type(self_id)],ELNAME(self_id));
            else 
             sprintf(errmsg,"Square root of negative number %g in %s.\n",*stacktop,ex_current.name);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(1263,errmsg,RECOVERABLE);
          }
        }
        else *stacktop = sqrt(*stacktop);
        break;

      case CEIL_NODE:
          *stacktop = ceil(*stacktop);
          break;

      case FLOOR_NODE:
          *stacktop = floor(*stacktop);
          break;

      case ABS_NODE:
         *stacktop = fabs(*stacktop);
        break;

      case SIN_NODE:
         *stacktop = sin(*stacktop);
        break;

      case COS_NODE:
         *stacktop = cos(*stacktop);
        break;

      case TAN_NODE:
         *stacktop = tan(*stacktop);
        break;

      case EXP_NODE:
         *stacktop = exp(*stacktop);
        break;

      case SINH_NODE:
         *stacktop = (exp(*stacktop)-exp(-*stacktop))/2;
        break;

      case COSH_NODE:
         *stacktop = (exp(*stacktop)+exp(-*stacktop))/2;
        break;

      case TANH_NODE:
         y = exp(*stacktop);
         *stacktop = (y*y-1)/(y*y+1) ;
        break;

      case ASINH_NODE:
         *stacktop = log(*stacktop + sqrt(*stacktop*(*stacktop) + 1));
         break;

      case ACOSH_NODE:
         if ( *stacktop < 1.0 )
         { if ( valid_id(self_id) )
             sprintf(errmsg,"Acosh argument less than 1 in %s, %s %s.\n",
               ex_current.name,typenames[id_type(self_id)],ELNAME(self_id));
           else sprintf(errmsg,"Acosh argument less than 1 in %s.\n",ex_current.name);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2557,errmsg,RECOVERABLE);
         }
         *stacktop = 2*log(sqrt(*stacktop+1) + sqrt(*stacktop - 1)) - log(2.0);
         break;

      case ATANH_NODE:
         if ( fabs(*stacktop) >= 1.0 )
         { if ( valid_id(self_id) )
             sprintf(errmsg,
               "Atanh argument magnitude not less than 1 in %s, %s %s.\n",
                ex_current.name,typenames[id_type(self_id)],ELNAME(self_id));
           else
            sprintf(errmsg,"Atanh argument magnitude not less than 1 in %s.\n",
              ex_current.name);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2559,errmsg, RECOVERABLE);
         }
         *stacktop = log(*stacktop+1)/2 - log(1-*stacktop)/2; 
         break;

      case LOG_NODE:
         if ( *stacktop <= 0.0 )
         { if ( valid_id(self_id) )
             sprintf(errmsg,"Log of zero or negative number in %s, %s %s.\n",
               ex_current.name,typenames[id_type(self_id)],ELNAME(self_id));
           else sprintf(errmsg,"Log of zero or negative number in %s.\n",
              ex_current.name);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(1264,errmsg, RECOVERABLE );
         }
         *stacktop = log(*stacktop);
        break;

      case ASIN_NODE:
         if ( *stacktop > 1.0 ) *stacktop = asin(1.0);
         else if ( *stacktop < -1.0 ) *stacktop = asin(-1.0);
         else *stacktop = asin(*stacktop);
        break;

      case ACOS_NODE:
         if ( *stacktop > 1.0 ) *stacktop = 0.0;
         else if ( *stacktop < -1.0 ) *stacktop = M_PI;
         else *stacktop = acos(*stacktop);
        break;

      case ATAN_NODE:
         *stacktop = atan(*stacktop);
        break;
      
      case ELLIPTICK_NODE:
         *stacktop = ellipticK(*stacktop);
        break;

      case ELLIPTICE_NODE:
         *stacktop = ellipticE(*stacktop);
        break;

      case INCOMPLETE_ELLIPTICF_NODE:
        stacktop--;
         *stacktop = incompleteEllipticF(stacktop[0],stacktop[1]);
        break;

      case INCOMPLETE_ELLIPTICE_NODE:
        stacktop--;
         *stacktop = incompleteEllipticE(stacktop[0],stacktop[1]);
        break;

      case CHS_NODE:
         *stacktop = -*stacktop;
        break;
      
      case INV_NODE:
         if ( *stacktop == 0.0 )
         { if ( valid_id(self_id) )
             sprintf(errmsg,"Division by zero in %s, %s %s.\n",ex_current.name,
             typenames[id_type(self_id)],ELNAME(self_id));
           else sprintf(errmsg,"Division by zero in %s.\n",ex_current.name);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2560,errmsg,RECOVERABLE); 
         }
         *stacktop = 1/(*stacktop);
        break;

      /* here are attributes for queries */
      case COORD_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
        { case VERTEX:
             *++stacktop = get_coord(id)[node->op2.coordnum];
             break;
          case EDGE:
             get_edge_side(id,vect);
             *++stacktop = vect[node->op2.coordnum];
             break;
          case FACET:
             get_facet_normal(id,vect);
             *++stacktop = vect[node->op2.coordnum];
             break;
         }
        break;

      case INDEXED_COORD_NODE:
      { int k = (int)*stacktop - 1;  /* 1 based indexing */
        if ( k < 0 || k >= SDIM )
        { sprintf(errmsg,
         "Invalid index %d for x in %s; must be between 1 and %d, inclusive.\n",
            k+1,ex_current.name,SDIM);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2061,errmsg,RECOVERABLE );
        }
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch( id_type(id) )
         { case VERTEX:
             *stacktop = get_coord(id)[k];
             break;
          case EDGE:
             get_edge_side(id,vect);
             *stacktop = vect[k];
             break;
          case FACET:
             get_facet_normal(id,vect);
             *stacktop = vect[k];
             break;
          default: 
             sprintf(errmsg,"Can't have indexed x on %s, in %s.\n",
               typenames[id_type(id)], ex_current.name);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2062,errmsg,RECOVERABLE);
         }
       }
       break;

      case PRINT_VERTEXNORMAL_NODE:
         { MAT2D(normal,MAXCOORD,MAXCOORD);
           REAL mag;
           int i;
           int normcount;

           if ( node->op1.localnum ) 
             id = *(element_id*)get_localp(node->op1.localnum);
           else id = q_id;
           normcount = new_calc_vertex_normal(id,normal); 
        //   project_vertex_normals(id,normal,normcount);
           mag = sqrt(SDIM_dot(normal[0],normal[0]));
           if ( mag == 0.0 ) 
           { mag = 1; memset(normal[0],0,SDIM*sizeof(REAL));}
           sprintf(msg,"{");
           for ( i = 0 ; i < SDIM ; i++ )
           { if ( i > 0 ) strcat(msg,",");
#ifdef FLOAT128
             sprintf(msg+strlen(msg),"%#*.*Qf",DWIDTH,DPREC,normal[0][i]/mag); 
#elif defined(LONGDOUBLE)
             sprintf(msg+strlen(msg),"%#*.*Lf",DWIDTH,DPREC,normal[0][i]/mag); 
#else
             sprintf(msg+strlen(msg),"%17.15g",normal[0][i]/mag); 
#endif
           }
           strcat(msg,"}\n");
           outstring(msg);
           break;
         }

      case GET_VERTEXNORMAL_NODE:     
         { MAT2D(normal,MAXCOORD,MAXCOORD);
           int k = (int)*stacktop - 1;  /* 1 based indexing */
           REAL mag;
           int normcount;

           if ( k < 0 || k >= SDIM )
             { sprintf(errmsg,
        "Invalid index %d for vertexnormal in %s; must be between 1 and %d.\n",
                 k+1,ex_current.name,SDIM);
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
               kb_error(2063,errmsg,RECOVERABLE );
             }
           if ( node->op1.localnum ) 
             id = *(element_id*)get_localp(node->op1.localnum);
           else id = q_id;
           normcount = new_calc_vertex_normal(id,normal); 
      //     project_vertex_normals(id,normal,normcount);
           mag = sqrt(SDIM_dot(normal[0],normal[0]));
           *stacktop = mag == 0.0 ? 0.0 : normal[0][k]/mag;
         }
        break;

      case PARAM_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
         *++stacktop = get_param(id)[node->op2.coordnum];
        break;


      case GET_SQ_MEAN_CURV_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
         *++stacktop = vertex_sq_mean_curvature(id);
        break;

      case GET_FIXEDVOL_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
         *++stacktop = get_battr(id)&FIXEDVOL ? 1.0 : 0.0;
        break;

      case GET_CENTEROFMASS_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
         *++stacktop = get_battr(id)&WANT_CENTEROFMASS ? 1.0 : 0.0;
        break;

      case GET_MEANCURV_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
         *++stacktop = vertex_mean_curvature(id);
        break;

      case GET_LENGTH_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( web.representation != STRING ) calc_edge(id);
         *++stacktop = get_edge_length(id);
        break;

      case GET_DIHEDRAL_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( id_type(id) == EDGE ) *++stacktop = dihedral(id);
        else if ( id_type(id) == VERTEX ) *++stacktop = vertex_angle(id);
        else *++stacktop = 0.0;
        break;

      case GET_ORIENTATION_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & NEGBOUNDARY) ? -1.0 : 1.0;
        break;

      case VALENCE_NODE:
      case GET_VALENCE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
         { case VERTEX:
              if ( web.representation == SIMPLEX )
                *++stacktop = (REAL)get_vertex_fvalence(id);
              else 
                *++stacktop = (REAL)get_vertex_evalence(id);
              break;
            case EDGE:
              *++stacktop = (REAL)get_edge_valence(id);
              break;
            case FACET:
              *++stacktop = (REAL)get_facet_valence(id);
              break;
            case BODY:
              *++stacktop = (REAL)get_body_valence(id);
              break;
         }
        break;

      case GET_EDGE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
         *++stacktop = (REAL)(ordinal(get_fe_edge(id))+1);
        break;

      case GET_FACET_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
         *++stacktop = (REAL)(ordinal(get_fe_facet(id))+1);
        break;

      case AREA_NODE:
      case GET_AREA_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = get_facet_area(id);
        break;

      case GET_MID_EDGE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = get_vattr(id) & (Q_MIDEDGE|Q_MIDPOINT) ? 1.0 : 0.0; 
        break;

      case GET_MID_FACET_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = get_vattr(id) & Q_MIDFACET ? 1.0 : 0.0; 
        break;

      case GET_WRAP_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = web.symmetry_flag ? (REAL)get_edge_wrap(id) : 0; 
        break;

      case GET_PRESSURE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
          { 
             case BODY: *++stacktop = get_body_pressure(id); break;
             default: 
              sprintf(errmsg,"Pressure only for bodies, in %s.\n",ex_current.name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(1267,errmsg,RECOVERABLE);

          }
        break;

      case GET_QUANTITY_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( !valid_id(id) )
        { sprintf(errmsg,
           "Quantity name '%s' needs attribute like .value (in %s)\n",
               GEN_QUANT(node->op2.quant_id)->name,ex_current.name);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2064,errmsg, RECOVERABLE);
        }
        *++stacktop = quantity_attribute(id,node->op2.quant_id);
        break;

      case GET_INSTANCE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( !valid_id(id) )
        { sprintf(errmsg,
           "Instance name '%s' needs attribute like .value (in %s)\n",
             METH_INSTANCE(node->op2.meth_id)->name,ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2065,errmsg, RECOVERABLE);
        }
        *++stacktop = instance_attribute(id,node->op2.meth_id);
        break;

      case GET_PHASE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
          { 
             case FACET: *++stacktop = (REAL)get_f_phase(id); break;
             case BODY: *++stacktop = (REAL)get_b_phase(id); break;
             default: 
               sprintf(errmsg,"Phase of wrong type element in %s.\n",ex_current.name);
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
               kb_error(1268,errmsg,RECOVERABLE);
          }
        break;

      case DENSITY_NODE:
      case GET_DENSITY_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
        { case EDGE: *++stacktop = get_edge_density(id); break;
          case FACET: *++stacktop = get_facet_density(id); break;
          case BODY: *++stacktop = get_body_density(id); break;
          default: 
            sprintf(errmsg,"Density of wrong type element in %s.\n",ex_current.name);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(1269,errmsg,RECOVERABLE);
          }
        break;

      case VOLUME_NODE:
      case GET_VOLUME_NODE:
      { ATTR attr;
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        attr = get_battr(id);
        if ( attr & FIXEDVOL )
        { if (fixed_volume_timestamp < global_timestamp) 
            calc_content(Q_FIXED);
        }
        else 
          if (  (info_volume_timestamp < global_timestamp) )
           calc_content(Q_INFO|Q_ENERGY);
        *++stacktop = get_body_volume(id);
        break;
       }

      case GET_VOLCONST_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = get_body_volconst(id);
        break;

      case GET_TARGET_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = get_body_fixvol(id);
        break;

      case ID_NODE:
      case GET_ID_NODE:
      case GET_OID_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( (node->type == GET_OID_NODE) && inverted(id) )
          *++stacktop = -(REAL)(ordinal(id)+1);
        else *++stacktop = (REAL)(ordinal(id)+1);
        break;

      case ORIGINAL_NODE:
      case GET_ORIGINAL_NODE:  /* as user's element id number */
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = valid_id(id) ? (REAL)ordinal(get_original(id))+1 : 0;
        break;

      case GET_MPI_TASK_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = id_task(id);
        break;


      case GET_COLOR_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
         { case EDGE:  *++stacktop = (REAL)get_edge_color(id); break;
           case FACET: *++stacktop = (REAL)get_facet_color(id); break;
           default: *++stacktop = 0.0;
         }
        break;

      case GET_FRONTCOLOR_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
         { case FACET:  *++stacktop = (REAL)get_facet_frontcolor(id); break;
            default: *++stacktop = 0.0;
         }
        break;

      case GET_BACKCOLOR_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
         { case FACET:  *++stacktop = (REAL)get_facet_backcolor(id); break;
            default: *++stacktop = 0.0;
         }
        break;

      case GET_FRONTBODY_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
         { case EDGE: 
             fe = get_edge_fe(id);
             if ( !valid_id(fe) ) {*++stacktop = 0.0; break;}
             f_id = get_fe_facet(fe);
             if ( inverted(f_id) ) f_id = get_fe_facet(get_next_facet(fe));
             if ( inverted(f_id) ) { *++stacktop = 0.0; break;}
             *++stacktop = (REAL)ordinal(get_facet_body(f_id))+1; 
             break;
           case FACET:  
              *++stacktop = (REAL)ordinal(get_facet_body(id))+1; 
              break;
            default: *++stacktop = 0.0;
         }
        break;

      case GET_BACKBODY_NODE:
    if ( node->op1.localnum ) 
       id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch ( id_type(id) )
         { case FACET:  *++stacktop =
              (REAL)ordinal(get_facet_body(inverse_id(id)))+1;
            break;
           case EDGE: 
             fe = get_edge_fe(id);
             if ( !valid_id(fe) ) {*++stacktop = 0.0; break;}
             f_id = get_fe_facet(fe);
             if ( !inverted(f_id) ) f_id = get_fe_facet(get_next_facet(fe));
             if ( !inverted(f_id) ) { *++stacktop = 0.0; break;}
             *++stacktop = (REAL)ordinal(get_facet_body(inverse_id(f_id)))+1; 
             break;
            default: *++stacktop = 0.0;
         }
        break;

      case GET_BARE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & BARE_NAKED) ? 1.0 : 0.0;
        break;

      case GET_MIDV_NODE:
        if ( web.modeltype != QUADRATIC )
        { sprintf(errmsg,"Cannot do MIDV except in QUADRATIC model (in %s).\n",
              ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2066,errmsg,RECOVERABLE);
        }
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = ordinal(get_edge_midv(id)) + 1.0;
        break;

      case FIXED_NODE:
      case GET_FIXED_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( id_type(id) == BODY )
          *++stacktop = get_battr(id)&FIXEDVOL ? 1.0 : 0.0;
        else *++stacktop = (get_attr(id) & FIXED) ? 1.0 : 0.0;
        break;

      case GET_OPACITY_NODE:
             if ( !opacity_attr ) *(++stacktop) = 1.0;
             else
             { if ( node->op1.localnum ) 
                 id = *(element_id*)get_localp(node->op1.localnum);
               else id = q_id;
               *(++stacktop) = *(REAL*)(get_extra(id,opacity_attr));
             }
             break;


      case GET_NO_DISPLAY_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & NODISPLAY) ? 1.0 : 0.0;
        break;

      case GET_NONCONTENT_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & NONCONTENT) ? 1.0 : 0.0;
        break;

      case GET_HIT_PARTNER_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & HIT_PARTNER) ? 1.0 : 0.0;
        break;

      case GET_NO_REFINE_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & NO_REFINE) ? 1.0 : 0.0;
        break;

      case GET_NO_TRANSFORM_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & NO_TRANSFORM) ? 1.0 : 0.0;
        break;

      case GET_TRIPLE_PT_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & TRIPLE_PT) ? 1.0 : 0.0;
        break;

      case GET_TETRA_PT_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & TETRA_PT) ? 1.0 : 0.0;
        break;

      case GET_NO_HESSIAN_NORMAL_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & NO_HESSIAN_NORMAL_ATTR) ? 1.0 : 0.0;
        break;

      case GET_AXIAL_POINT_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        *++stacktop = (get_attr(id) & AXIAL_POINT) ? 1.0 : 0.0;
        break;

      case GET_SHOW_NODE:
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( id_type(id) == EDGE )
        { ATTR  eattr = get_eattr(id);
          int showflag = 0;
          facetedge_id fe_id = get_edge_fe(id);
 
          if ( eattr & BOUNDARY ) showflag = 1;
          if ( equal_id(get_next_facet(fe_id),fe_id) ) /* valence 1 */
              showflag = 1;
          else if  ( !equal_id(get_next_facet(fe_id),get_prev_facet(fe_id)) )
              showflag = 1;
          if ( eattr & HIT_WALL )  showflag = 1;
          if ( eattr & FIXED ) showflag = 1;
          if ( show_expr[EDGE] && show_expr[EDGE]->start )
          {
            PROF_EVAL_END(ex_profiling);
            PUSH_TRACE;
            showflag = eval(show_expr[EDGE],NULL,id,NULL) ? 1 : 0; 
            POP_TRACE;
            PROF_EVAL_START(ex_profiling);
          }
          if ( get_edge_color(id) == CLEAR ) showflag = 0;
          *++stacktop = showflag;
        }
        else if ( id_type(id) == FACET )
        { ATTR fattr = get_fattr(id);
          int showflag = 1;

          if ( (fattr & (BOUNDARY|CONSTRAINT)) && !bdry_showflag )
            showflag = 0;
          if ( fattr & NODISPLAY )
             showflag = 0;

          if ( no_wall_flag )
          { /* skip facets with all three vertices on walls */
            fe = get_facet_fe(id);
            if ( get_vattr(get_fe_headv(fe)) & (HIT_WALL|CONSTRAINT) )
            if ( get_vattr(get_fe_tailv(fe)) & (HIT_WALL|CONSTRAINT) ) 
            { fe = get_next_edge(fe);
              if ( get_vattr(get_fe_headv(fe)) & (HIT_WALL|CONSTRAINT) )
               showflag = 0;
            }
          }      

          if ( show_expr[FACET] && show_expr[FACET]->start )
          { 
            PROF_EVAL_END(ex_profiling);
            if ( !eval(show_expr[FACET],NULL,id,NULL) ) 
               showflag = 0;
            PROF_EVAL_START(ex_profiling);
          }
          *++stacktop = showflag;
         }
         else
         { sprintf(errmsg,
             "\"show\" attribute applied to wrong type of element in %s.\n",
                ex_current.name);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(2067,errmsg,RECOVERABLE);
         }
         break;

      case ATTR_FUNCTION_NODE:
           node += node->op1.skipsize - 1;
           break;
      case ATTR_FUNCTION_END_NODE:
         { struct extra *ext = EXTRAS(node->op2.eltype) + node->op1.extranum;
           ext->flags |= FUNCTION_ATTR;
           tree_copy(&ext->code,node + node->right);
           break;
         }

      case GET_EXTRA_ATTR_NODE:
       { struct extra *ext;
         int spot;
         n = node->op3.extranum; /* attribute number */
         if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
         else id = q_id;
         ext = EXTRAS(node->op2.eltype) + n;
         /* get index */
         spot = 0;
         for ( k = 0 ; k < ext->array_spec.dim ; k++ )
         { int j = (int)(stacktop[-ext->array_spec.dim+k+1]);
           spot *= ext->array_spec.sizes[k];
           if ( (j < 1) || (j > ext->array_spec.sizes[k]) )
           { sprintf(errmsg,
                "Attribute %s index %d is %d; maximum is %d (in %s).\n",
                 ext->name,k+1,j,ext->array_spec.sizes[k],ex_current.name);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1270,errmsg,RECOVERABLE);
           }
           spot += (int)(stacktop[-ext->array_spec.dim+k+1]) - 1;
         }
         stacktop -= ext->array_spec.dim;
         if ( id_type(id) != node->op2.eltype )
         { if ( (id_type(id)==EDGE) && (node->op2.eltype==VERTEX) && params )
           { ext = EXTRAS(VERTEX) + n;
             *++stacktop = interp_edge_attribute(id,ext,spot,(int)params[2*SDIM]);
             break;
           }
           else 
           if ( (id_type(id)==FACET) && (node->op2.eltype==VERTEX) && params )
           { ext = EXTRAS(VERTEX) + n;
             *++stacktop = interp_facet_attribute(id,ext,spot,(int)params[2*SDIM]);
             break;
           }
           else 
           { sprintf(errmsg,
               "Attribute %s is %s attribute, not %s attribute (in %s).\n",
               EXTRAS(node->op2.eltype)[n].name,
                 typenames[node->op2.eltype], typenames[id_type(id)],ex_current.name);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(2068,errmsg,RECOVERABLE);
           }
         }
         if ( ext->code.start ) 
         { int oldflag = autorecalc_flag;
           autorecalc_flag = 0;

           PROF_EVAL_END(ex_profiling);
           PUSH_TRACE;
           eval(&ext->code,NULL,id,NULL);  /* side-effect fills in values */
           POP_TRACE;
           PROF_EVAL_START(ex_profiling);
           autorecalc_flag = oldflag;
          }
         switch ( ext->type )
         { case REAL_TYPE: *++stacktop = ((REAL*)get_extra(id,n))[spot]; 
              break;
           case INTEGER_TYPE: 
           case CONSTRAINT_TYPE:
           case BOUNDARY_TYPE:
           case QUANTITY_TYPE:
           case INSTANCE_TYPE:
           case PROCEDURE_TYPE:
             *++stacktop = (REAL)((int*)get_extra(id,n))[spot];
              break;
           case UINT_TYPE: 
             *++stacktop = (REAL)((unsigned int*)get_extra(id,n))[spot];
              break;
           case ULONG_TYPE: 
             *++stacktop = (REAL)((unsigned long*)get_extra(id,n))[spot];
              break;
           case LONG_TYPE: 
             *++stacktop = (REAL)((long*)get_extra(id,n))[spot];
              break;
           case UCHAR_TYPE: 
             *++stacktop = (REAL)((unsigned char*)get_extra(id,n))[spot];
              break;
           case CHAR_TYPE: 
             *++stacktop = (REAL)((char*)get_extra(id,n))[spot];
              break;
           case SHORT_TYPE: 
             *++stacktop = (REAL)((short int*)get_extra(id,n))[spot];
              break;
           case USHORT_TYPE: 
             *++stacktop = (REAL)((unsigned short int*)get_extra(id,n))[spot];
              break;
           case ELEMENTID_TYPE: 
           case VERTEX_TYPE: 
           case EDGE_TYPE: 
           case FACET_TYPE: 
           case BODY_TYPE: 
           case FACETEDGE_TYPE: 
             *(element_id*)(++stacktop) = 
                ((element_id*)get_extra(id,n))[spot];
              break;
           case PTR_TYPE: 
             *((char**)++stacktop) =  (((char**)get_extra(id,n))[spot]);
              break;
         }
       }
       break; 

      case ON_CONSTRAINT_NODE:
      case ON_CONSTRAINT_NAME_NODE:
      { int testcon = (node->type == ON_CONSTRAINT_NODE) ? (int)*(stacktop--)
                            : node->op3.connum;
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch(id_type(id))
         { case VERTEX: *++stacktop = (REAL)v_on_constraint(id,testcon); break;
           case EDGE  : *++stacktop = (REAL)e_on_constraint(id,testcon); break;
           case FACET : *++stacktop = (REAL)f_on_constraint(id,testcon); break;
           default: 
              sprintf(errmsg,
                 "Can't do constraints on this type element (in %s).\n",
                     ex_current.name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(1272,errmsg,RECOVERABLE);
         }
       }
      break;

      case CONSTRAINT_VALUE_NODE:
      case CONSTRAINT_NAME_VALUE_NODE:
      { int connum = (node->type == CONSTRAINT_VALUE_NODE) ? (int)*(stacktop--)
                            : node->op3.connum;
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        if ( id_type(id) != VERTEX )
        { sprintf(errmsg,"Can only do constraint value for vertices.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(4272,errmsg,RECOVERABLE);
        }
        PUSH_TRACE;
        *++stacktop = eval(get_constraint(connum)->formula ,get_coord(id), id, NULL);
        POP_TRACE;
      }
      break;

      case HIT_CONSTRAINT_NODE:
      case HIT_CONSTRAINT_NAME_NODE:
      { int testcon = (node->type == HIT_CONSTRAINT_NODE) ? (int)*(stacktop--)
                            : node->op3.connum;
        if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
        else id = q_id;
        switch(id_type(id))
        { case VERTEX: *++stacktop = (REAL)get_v_constraint_status(id,testcon); 
              break;
          default: 
             sprintf(errmsg, 
                "Can do hit_constraints only on vertices (in %s).\n",ex_current.name);
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1273,errmsg,RECOVERABLE);
        }
      }
      break;

      case ON_BOUNDARY_NODE:
      case ON_BOUNDARY_NAME_NODE:
       { struct boundary *b=NULL;
         int testb = (node->type == ON_BOUNDARY_NODE) ? (int)*(stacktop--)
                            : node->op3.bdrynum;
         if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
         else id = q_id;
         if ( !(get_attr(id) & BOUNDARY) )
         { *++stacktop = 0.0;
           break;
         }
         switch(id_type(id))
         { case VERTEX: b = get_boundary(id); break;
           case EDGE  : b = get_edge_boundary(id); break;
           case FACET : b = get_facet_boundary(id); break;
           default: 
              sprintf(errmsg,
                 "Can't do boundary on a %s (in %s).\n",
                    typenames[id_type(id)],ex_current.name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(1274,errmsg,RECOVERABLE);
         }
         *++stacktop = (b == web.boundaries+testb) ? 1.0 : 0.0;
       }
       break;

      case ON_METHOD_INSTANCE_NODE:
       { struct element *eptr;
         int *mptr;
         if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
         else id = q_id;
         eptr = elptr(id);
         mptr = (int*)((char*)eptr+get_meth_offset(id_type(id)));
         for ( i = 0 ; i < eptr->method_count ; i++ )
           if ( abs(node->op2.meth_id) == abs(mptr[i]) )
           { *++stacktop = 1.0;
             break;
           }
         if ( i == eptr->method_count ) 
           *++stacktop = 0.0;
         break;
       } 

      case ON_QUANTITY_NODE:
       { struct element *eptr;
         int *mptr;
         int found = 0;
         int j;
         struct method_instance *mi;
         if ( node->op1.localnum ) 
           id = *(element_id*)get_localp(node->op1.localnum);
         else id = q_id;
         eptr = elptr(id);
         mptr = (int*)((char*)eptr+get_meth_offset(id_type(id)));
         for ( i = 0 ; i < eptr->method_count ; i++ )
         { mi = METH_INSTANCE(abs(mptr[i]));
           for ( j = 0 ; j < MMAXQUANTS ; j++ )
             if ( mi->quants[j] == node->op2.quant_id )
             { *++stacktop = 1.0;
               found = 1;
               break;
             }
         }
         if ( !found ) 
           *++stacktop = 0.0;
         break;
       } 

      case SELF_ELEMENT_NODE:
        if ( !valid_id(self_id) )
        { sprintf(errmsg,"No element for SELF to refer to in %s.\n",ex_current.name);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2069,errmsg,RECOVERABLE);
        }
        *(element_id*)get_localp(node->op2.localnum) = self_id;
        break;

      case SYMBOL_ELEMENT_NODE:
        break;

      case SINGLE_ELEMENT_EXPR_NODE:
          *(element_id*)(++stacktop) = 
              *(element_id*)get_localp(node[node->left].op2.localnum);
          break;

      case ELEMENT_IDENT_NODE:
          *(element_id*)get_localp(node->op2.localnum)
             = globals(node->op3.name_id)->value.id;
          break;

      case INDEXED_SUBTYPE_NODE:  /* like ee.vertex[1] */
        { element_id next_id;
          int ord = (int)*(stacktop--) - 1;  /* which one */
          element_id parent = node[node->left].op2.localnum ?
            *(element_id*)get_localp(node[node->left].op2.localnum) : q_id;
          next_id = get_indexed_subtype(node,ord,parent,&ex_current);
          *(element_id*)get_localp(node->op2.localnum) = next_id;
        } break;  /* end INDEXED_SUBTYPE */

      case INDEXED_ELEMENT_NODE:
       { element_id partid = *(element_id*)(stacktop--);
         element_id id;
         id = get_full_id(node->op1.eltype,partid);
         if ( !valid_id(id) ) 
         { sprintf(errmsg,"%s index %d is not valid.\n",
              typenames[node->op1.eltype],
              valid_id(partid) ? (int)(partid & OFFSETMASK)+1 : 0);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(1200,errmsg,RECOVERABLE);
         }
         *(element_id *)get_localp(node->op2.localnum) = id;
         break;
       }

      case QUALIFIED_ATTRIBUTE_NODE:
          break; /* just a no-op in execution */

         /****************************/
         /* aggregate initialization */
         /****************************/

      case SET_INIT_NODE: 
        web.where_count = 0;
        /* break and continue jumps */
        *(size_t*)(newstack + localstack + node->stackpos) = stacktop - (newstack + localstack) + 3;
#ifdef MPI_EVOLVER
        if ( this_task == MASTER_TASK ) 
        { mpi_subtask_command_flag = 1;
          mpi_aggregate(node,&ex_current);
        }
#endif
        break;

      case AGGREGATE_INIT_NODE:
        aggregate_depth++;
        web.where_count = 0;
        /* break and continue jumps */
        *(size_t*)(newstack + localstack + node->stackpos) = stacktop - (newstack + localstack) + 3; 
        switch ( node->op1.aggrtype )
          {  case FOREACH_NODE:
             case SET_ATTRIBUTE_LOOP_NODE:
                break;

              case MAX_NODE:
                *++stacktop = -MAXDOUBLE; /* max of empty set */
                break;

              case MIN_NODE:
                *++stacktop = MAXDOUBLE; /* min of empty set */
                break;

              case SUM_NODE: 
                *++stacktop = 0.0; /* sum */
                break;

              case AVG_NODE:
                *++stacktop = 0.0; /* count */
                *++stacktop = 0.0; /* sum */
                break;

              case COUNT_NODE:
                *++stacktop = 0.0; /* count */
                break;

              case HISTOGRAM_NODE:
              case LOGHISTOGRAM_NODE:
                *++stacktop = 0.0; /* first phase counter */
                *++stacktop = MAXDOUBLE;  /* min */
                *++stacktop = -MAXDOUBLE; /* max */
                for ( n = 0 ; n < HISTBINS+1 ; n++ )
                  *++stacktop = 0.0;
                histo_max = web.skel[node->op2.eltype].count + 5;
                histo_data = (REAL*)temp_calloc(histo_max,sizeof(REAL));
                histo_count = 0;
                break;

              case LIST_NODE: /* print column headers */
#ifdef MPI_EVOLVER
               if ( (this_task == MASTER_TASK) || !mpi_subtask_command_flag )
#endif
                switch ( node->op2.eltype )
                { case VERTEX:
                  if ( SDIM == 2 )
     outstring("// Id               X                Y\n");
                else
     outstring("// Id               X                Y               Z\n");
                    break;
                 case EDGE:
                    outstring("// Id    endpoints\n");
                    break;
                 case FACET:
                    if ( web.representation == SIMPLEX )
                      outstring("// Id      vertices\n");
                    else outstring("// Id    edges\n");
                    break;
                 case BODY:
                    outstring("// Id    facets\n");
                    break;
                 case FACETEDGE:
                    outstring("//    id       edge    facet  prevedge nextedge    prevfacet nextfacet\n");
                    break;
                }
              break;

              case REFINE_NODE: 
#ifdef MPI_EVOLVER
               if ( (this_task != MASTER_TASK) && mpi_subtask_command_flag 
                 &&  node->op2.eltype == EDGE )
                     mpi_edge_refine_init();
#endif
                break;
              
              case DISSOLVE_NODE:
                break;

              case DELETE_NODE:
#ifdef MPI_EVOLVER
               if ( this_task == MASTER_TASK )
                 mpi_set_corona(THIN_CORONA);

               if ( (this_task != MASTER_TASK) && mpi_subtask_command_flag )
                     mpi_delete_init();
#endif
                break;
              
              case FIX_NODE:
                break;

              case UNFIX_NODE:
                break;
              case REVERSE_ORIENTATION_NODE: break;
              case EDGESWAP_NODE: break;
              case T1_EDGESWAP_NODE: break;
              case EQUIANGULATE_NODE: break;

              case POP_NODE: case POP_TRI_TO_EDGE_NODE:
              case POP_EDGE_TO_TRI_NODE: case POP_QUAD_TO_QUAD_NODE:
                break;

              case VERTEX_AVERAGE_NODE: break;
              case RAW_VERTEX_AVERAGE_NODE: break;
              case RAWEST_VERTEX_AVERAGE_NODE: break;

              default: 
                 sprintf(errmsg,"Internal error: Bad aggregate type %d.\n",
                      node->op1.aggrtype);
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(1290,errmsg, RECOVERABLE);
                 break;
            }
#ifdef MPI_EVOLVER
                if ( this_task == MASTER_TASK ) 
                { mpi_subtask_command_flag = 1;
                  mpi_aggregate(node,&ex_current);
                }
#endif

         break;
         /* end case AGGREGATE_INIT */

        /************************/
        /* next element in loop */
        /************************/
         
      case SINGLE_ELEMENT_NODE: 
          q_id = *(element_id*)get_localp(node[node->left].op2.localnum);
          if ( stacktop[-1] > 0.0 ) /* done */
              node += node->op1.skipsize;
          else stacktop[-1] = 1.0;
          break;

      case NEXT_VERTEX_NODE: /* all vertices */
       { 
         vertex_id *vp;
         vp = (element_id *)(stacktop - 0);
         if ( !valid_id(*vp) || breakflag )
         { /* done */ 
           node += node->op1.skipsize - 1;
           break;
         }
         q_id = *vp;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         /* check sentinel; set for next time around */
         if ( equal_id(*vp,*(vertex_id*)(stacktop-1)) ) *vp = NULLID;
         else *vp = vptr(*vp)->forechain;

         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign vertices in global aggregate */
#endif
        }
        break; 

      case NEXT_EDGE_VERTEX_NODE: /* edge vertices */
       {
         int *numptr;
         vertex_id *vp;
         numptr = (int *)(stacktop - 1);
         if ( *numptr== web.skel[EDGE].ctrlpts ) 
            { /* done */
              node += node->op1.skipsize - 1;
              break;
            }
         vp = *(element_id **)(stacktop - 0);
         q_id = vp[*numptr];
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         (*numptr)++;

          if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
         break;
       }

      case NEXT_FACET_VERTEX_NODE: /* facet vertices */
        {

          facetedge_id *fe_ptr;
          facet_id *f_ptr;
          if ( (web.representation == SIMPLEX) ||
              (web.modeltype==LAGRANGE) )
          { int *numptr = (int *)(stacktop - 1);
            if ( *numptr > web.skel[FACET].ctrlpts )
            { node += node->op1.skipsize - 1;
              break;
            }
            f_ptr = (facet_id *)(stacktop);
            q_id = get_facet_vertices(*f_ptr)[*numptr];
            (*numptr)++;
          }
          else if ( (web.modeltype == QUADRATIC) 
                      && (web.representation == SOAPFILM) )
          { int *numptr = (int *)(stacktop - 1);
            if ( *numptr >= web.skel[FACET].ctrlpts ) /* see if done */
            { node += node->op1.skipsize - 1;
              break;
            }
            f_id = *(facet_id *)(stacktop);
            fe = get_facet_fe(f_id);
            if ( *numptr < 3 )
            { for ( i = 0 ; i < *numptr ; i++ )
                fe = get_next_edge(fe);
              q_id = get_fe_tailv(fe);
            }
            else 
            { for ( i = 3 ; i < *numptr ; i++ )
                fe = get_next_edge(fe);
              q_id = get_fe_midv(fe);
            }
            (*numptr)++;
          }
          else
          { fe_ptr = (element_id *)(stacktop - 0);
            if ( !valid_id(*fe_ptr) ) /* see if done */
            { node += node->op1.skipsize - 1;
              break;
            }
            if ( id_type(*fe_ptr) == VERTEX )
            { /* last vertex in string unclosed facet */
              q_id = *fe_ptr; *fe_ptr = NULLID;
            }
            else
            { edge_id e_id = get_fe_edge(*fe_ptr);
              q_id = get_edge_tailv(e_id);
              *fe_ptr = get_next_edge(*fe_ptr);
              if ( !valid_id(*fe_ptr) ) 
                *fe_ptr = get_edge_headv(e_id); /* end of string facet */
              else if ( !valid_element(*fe_ptr) 
                    || equal_id(*fe_ptr,*(element_id *)(stacktop-1)))
                *fe_ptr = NULLID; /* last one */
            }
         }
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
         break;
        }

      case NEXT_BODY_VERTEX_NODE: /* body vertices */   
        break;

      case NEXT_EDGE_NODE: /* all edges */
       { 
         edge_id *ep; 
         ep = (element_id *)(stacktop - 0);
         if ( !valid_id(*ep) || breakflag )
         {  /* done */
            node += node->op1.skipsize - 1;
            break;
         }
         q_id = *ep;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         /* check sentinel; set for next time around */
         if ( equal_id(*ep,*(edge_id*)(stacktop-1)) ) *ep = NULLID;
         else  *ep = eptr(*ep)->forechain; 

         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
       }
       break;

      case NEXT_VERTEX_EDGE_NODE: /* vertex edges */
       { edge_id *e_ptr;

         e_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*e_ptr) ) /* see if done */
         { node += node->op1.skipsize - 1;
           break;
         }
         q_id = *e_ptr;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         if ( get_attr(*(element_id *)(stacktop-2)) & (Q_MIDPOINT|Q_MIDEDGE) )
            *e_ptr = NULLID; /* last one */
         else
         { *e_ptr = get_next_tail_edge(*e_ptr);
           if ( !valid_element(*e_ptr)
                  || equal_id(*e_ptr,get_vertex_edge(get_edge_tailv(*e_ptr))))
           *e_ptr = NULLID; /* last one */
         }

         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
         break;
       }

      case NEXT_FACET_EDGE_NODE: /* facet edges */
       { 
         facetedge_id *fe_ptr;
         fe_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*fe_ptr) ) /* see if done */
         { node += node->op1.skipsize - 1;
           break;
         }
         q_id = get_fe_edge(*fe_ptr);
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         *fe_ptr = get_next_edge(*fe_ptr);
         if ( web.representation == STRING )
         {
           if ( !valid_element(*fe_ptr) 
               || equal_id(*fe_ptr,*(element_id *)(stacktop-1)))
            *fe_ptr = NULLID; /* last one */
         }
         else
         { if ( ++(*(int*)(stacktop-1)) == FACET_VERTS )
           *fe_ptr = NULLID; /* last one */
         }
         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
         break;
       }

      case NEXT_BODY_EDGE_NODE: /* body edges */
        break;

      case NEXT_FACET_NODE:  /* all facets */
       { 
         facet_id *fp; 
         fp = (element_id *)(stacktop - 0);
         if ( !valid_id(*fp)|| breakflag  )
         { node += node->op1.skipsize - 1; /* skip to end */
           break;
         }
         q_id = *fp;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         /* check sentinel; set for next time around */
         if ( equal_id(*fp,*(facet_id*)(stacktop-1)) ) *fp = NULLID;
         else *fp = fptr(*fp)->forechain;
         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
       }
       break;

      case NEXT_EDGE_FACET_NODE:  /* all facets on edge */
       { 
         facetedge_id *fe_ptr;
         fe_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*fe_ptr) ) /* see if done */
         { node += node->op1.skipsize - 1;
           break;
         }
         q_id = get_fe_facet(*fe_ptr);
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         *fe_ptr = get_next_facet(*fe_ptr);
         if ( !valid_element(*fe_ptr) || 
                  equal_element(*fe_ptr,get_edge_fe(get_fe_edge(*fe_ptr))))
                  *fe_ptr = NULLID; /* last one */

         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
         break;
       }

      case NEXT_EDGE_FACETEDGE_NODE:  /* all facetedges on edge */
       { 
         facetedge_id *fe_ptr;
         fe_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*fe_ptr) ) /* see if done */
         { node += node->op1.skipsize - 1;
           break;
         }
         q_id = *fe_ptr;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         *fe_ptr = get_next_facet(*fe_ptr);
         if ( !valid_element(*fe_ptr) || 
                  equal_element(*fe_ptr,get_edge_fe(get_fe_edge(*fe_ptr))))
                  *fe_ptr = NULLID; /* last one */

         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
         break;
       }

      case NEXT_VERTEX_FACET_NODE: /* facets adjacent to vertex */

       { 
         facet_id *f_ptr,first_f;
         f_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*f_ptr) ) /* see if done */
         { 
           node += node->op1.skipsize - 1;
           break;
         }
         q_id = *f_ptr;
         if ( id_type(q_id) == FACETEDGE )
           q_id = get_fe_facet(q_id);
         q_id = positive_id(q_id);
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         v_id = *(vertex_id*)(stacktop-2);
         *f_ptr=get_next_vertex_facet(v_id,*f_ptr);
         first_f = get_vertex_first_facet(v_id); /* might have changed */
         if ( equal_id(*f_ptr,first_f) )
           *f_ptr = NULLID;
         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
        }
        break;

      case NEXT_BODY_FACET_NODE: /* facets on body */
       { 
         facet_id *f_ptr;
         f_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*f_ptr) ) /* see if done */
         { node += node->op1.skipsize - 1;
           break;
         }
         q_id = *f_ptr;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         *f_ptr = get_next_body_facet(*f_ptr);
         if ( (get_fattr(*f_ptr)& (inverted(*f_ptr)?DID_BODYBACKFACET:DID_BODYFRONTFACET))
               || equal_id(*f_ptr,q_id) || 
                       equal_id(*f_ptr,*(element_id *)(stacktop-1)))
            *f_ptr = NULLID; /* last one */
         else set_attr(*f_ptr,
                (inverted(*f_ptr) ? DID_BODYBACKFACET : DID_BODYFRONTFACET) );
         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
 #ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
      }

       break;

      case NEXT_BODY_NODE: /* all bodies */
       { 
         body_id *bp; 
         bp = (element_id *)(stacktop - 0);
         if ( !valid_id(*bp)|| breakflag  )
         { node += node->op1.skipsize - 1;
           break;
         }  
         q_id = *bp;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         /* check sentinel; set for next time around */
         if ( equal_id(*bp,*(body_id*)(stacktop-1)) ) *bp = NULLID;
         else *bp = bptr(*bp)->forechain;

         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
       }
         break;

         case NEXT_FACET_BODY_NODE: /* both bodies on facet */
         { 
            int *numptr = (int *)(stacktop - 1);
            f_id = *(facet_id *)(stacktop - 2); /* facet */
            (*numptr)++;
            if ( *numptr == 3 ) /* see if done */
            { /* done */
              node += node->op1.skipsize - 1;
              break;
            }
            if ( *numptr == 1 ) /* first */
            { 
              q_id = get_facet_body(f_id);
              *(element_id*)get_localp(node->op2.localnum) = q_id;
              if ( !valid_id(q_id) )
                node--;  /* try again */
             }
             else /* *numptr == 2 */
             { q_id = get_facet_body(inverse_id(f_id));
               *(element_id*)get_localp(node->op2.localnum) = q_id;
               if ( !valid_id(q_id) )
               node--;  /* try again */
             }

             break;
         }

      case NEXT_VERTEX_BODY_NODE: /* bodies adjacent to vertex */
        break;
      case NEXT_EDGE_BODY_NODE: /* bodies adjacent to edge */
        break;

      case NEXT_FACETEDGE_NODE: /* all facetedges */
       { 
         facetedge_id *fep;
         fep = (element_id *)(stacktop - 0);
         if ( !valid_id(*fep)|| breakflag  )
         { /* done */ 
           node += node->op1.skipsize - 1;
           break;
         }
         q_id = *fep;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         /* check sentinel; set for next time around */
         if ( equal_id(*fep,*(facetedge_id*)(stacktop-1)) ) *fep = NULLID;
         else *fep = feptr(*fep)->forechain;

          if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
        }
        break; 


      case NEXT_FACETEDGE_EDGE_NODE: /* edge on facetedge */
       { 
         edge_id *e_ptr;
         e_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*e_ptr) ) /* see if done */
         { node += node->op1.skipsize - 1;
           break;
         }
         q_id = *e_ptr;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         *e_ptr = NULLID; /* last one */
         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
 #ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
      }
       break;


      case NEXT_FACETEDGE_FACET_NODE: /* facet on facetedge */
       { 
         facet_id *f_ptr;
         f_ptr = (element_id *)(stacktop - 0);
         if ( !valid_id(*f_ptr) ) /* see if done */
         { node += node->op1.skipsize - 1;
           break;
         }
         q_id = *f_ptr;
         *(element_id*)get_localp(node->op2.localnum) = q_id;
         *f_ptr = NULLID; /* last one */
         if ( !valid_element(q_id) ) node--;    /* skip body of loop */
#ifdef MPI_EVOLVER
         if ( mpi_subtask_command_flag && (id_task(q_id) != this_task) )
            node--;  /* don't count foreign elements in global aggregate */
#endif 
       }
       break;

        /***********************/
        /* aggregate condition */
        /***********************/

      case WHERE_NODE: /* see if element qualifies */
          if ( *(stacktop--) == 0.0 )
          node += node->left-1;  /* no, go back to generator */
          else node->op1.wherecount++; /* where count   NONREENTRANT */
          break;  /* else continue with current element */

      case SINGLE_ASSIGN_NODE:
          /* restore old q_id */
          q_id = *(element_id *)(stacktop - 2);
          stacktop -= 3;  /* erase locals */
          break;


        /*****************/
        /* aggregate end */
        /*****************/

      case AGGREGATE_END_NODE:
          aggregate_depth--;
          /* restore old q_id */
          q_id = *(element_id *)(stacktop - 2);
          where = node+node->right;
          where = where+where->left;
          if ( where->type == WHERE_NODE )
          { web.where_count = where->op1.wherecount;
            where->op1.wherecount = 0; /* reset where count */
          }
          switch ( node->op1.aggrtype )
          {
            case FOREACH_NODE:
            case SET_ATTRIBUTE_LOOP_NODE:
              stacktop -= 3;  /* erase locals */
              break; 

            case MAX_NODE:  
              stacktop -= 3;  /* erase locals */
#ifdef MPI_EVOLVER
              if ( mpi_subtask_command_flag && (aggregate_depth==0))
              { REAL total;
                MPI_Reduce(stacktop,&total,1,MPI_REAL,MPI_MAX,
                    MASTER_TASK,MPI_COMM_WORLD);
                if ( this_task == MASTER_TASK )
                  stacktop[0] = total;
              }
#endif
              break;

            case MIN_NODE: 
              stacktop -= 3;  /* erase locals */
#ifdef MPI_EVOLVER
              if ( mpi_subtask_command_flag && (aggregate_depth==0))
              { REAL total;
                MPI_Reduce(stacktop,&total,1,MPI_REAL,MPI_MIN,
                    MASTER_TASK,MPI_COMM_WORLD);
                if ( this_task == MASTER_TASK )
                  stacktop[0] = total;
              }
#endif
              break;

            case SUM_NODE: 
              stacktop -= 3;  /* erase locals */
#ifdef MPI_EVOLVER
              if ( mpi_subtask_command_flag && (aggregate_depth==0))
              { REAL total;
                MPI_Reduce(stacktop,&total,1,MPI_REAL,MPI_SUM,
                    MASTER_TASK,MPI_COMM_WORLD);
                if ( this_task == MASTER_TASK )
                  stacktop[0] = total;
              }
#endif
              break;

            case COUNT_NODE:
              stacktop -= 3;  /* erase locals */
#ifdef MPI_EVOLVER
              if ( mpi_subtask_command_flag && (aggregate_depth==0))
              { REAL total;
                MPI_Reduce(stacktop,&total,1,MPI_REAL,MPI_SUM,
                    MASTER_TASK,MPI_COMM_WORLD);
                if ( this_task == MASTER_TASK )
                  stacktop[0] = total;            
              }
#endif
              break;

            case AVG_NODE:
              stacktop -= 4;  /* erase locals */
#ifdef MPI_EVOLVER
              if ( mpi_subtask_command_flag && (aggregate_depth==0) )
              { REAL total[2];
                MPI_Reduce(stacktop,total,2,MPI_REAL,MPI_SUM,
                    MASTER_TASK,MPI_COMM_WORLD);
                if ( this_task == MASTER_TASK )
                { stacktop[0] = total[0];
                  stacktop[1] = total[1];
                }
              }
#endif
              if ( stacktop[0] > 0.0 )
                 stacktop[0] = stacktop[1]/stacktop[0];
              /* else leave avg as 0 */
              break;

             case LOGHISTOGRAM_NODE:
             case HISTOGRAM_NODE:
              { int i;
                REAL binsize;
                int one_bin_flag = 0;

                bins = stacktop - (HISTBINS+1) + 1 - 3;

                /* find hi and lo */
                hi = -MAXDOUBLE; lo = MAXDOUBLE;
                for ( i = 0 ; i < histo_count ; i++ )
                { if ( histo_data[i] < lo ) 
                    if ( (node->op1.aggrtype == HISTOGRAM_NODE) || ( histo_data[i] > 0.0) )
                      lo = histo_data[i];
                  if ( histo_data[i] > hi ) hi = histo_data[i];
                }
#ifdef MPI_EVOLVER
               if ( mpi_subtask_command_flag && (aggregate_depth==0))
                { /* synchronize max-min values */
                  DOUBLE localhi = hi,locallo = lo;
                  DOUBLE globalhi,globallo;
                  MPI_Allreduce(&localhi,&globalhi,1,MPI_REAL,MPI_MAX,
                      MPI_COMM_WORLD);
                  hi = globalhi;
                  MPI_Allreduce(&locallo,&globallo,1,MPI_REAL,MPI_MIN,
                      MPI_COMM_WORLD);
                  lo = globallo;
                }
#endif
             if (node->op1.aggrtype == HISTOGRAM_NODE )
             { if ( hi == lo )
               { binsize = 1.0; /* will put everything in first bin */
                 one_bin_flag = 1;
               }
               else binsize = (hi-lo)/HISTBINS;/* binsize */
             }
             else /* first bin for 0 values for log histogram */
              { if ( hi <= 0.0 ) binsize = 0.0;
                else 
                { hi = log(hi);
                  if ( lo <= 0.0 ) 
                  { binsize = 1.0; lo = hi-(HISTBINS-1); }
                  else 
                  { lo = log(lo); 
                    if ( hi == lo ) 
                    { binsize = 1.0;
                      one_bin_flag = 1;
                    }
                    else
                      binsize = (hi-lo)/(HISTBINS-1);
                  }
                }
              }
             binsize *= 1+10*machine_eps; /* for rounding margin */
             if ( lo != 0.0 ) lo -= 10*machine_eps*binsize;
             /* construct histogram */
             for ( i = 0 ; i < histo_count ; i++ )
             { val = histo_data[i];
                if ( !is_finite(val) ) 
                    k = HISTBINS;  /* NaN */ 
                else if ( node->op1.aggrtype == HISTOGRAM_NODE )
                  k = (int)((val-lo)/binsize);
                else if ( val <= 0.0 ) k = 0;
                else
                 { val = log(val);
                   k = (int)((val-lo)/binsize)+1;
                 }
                bins[k] += 1.0;
             }
#ifdef MPI_EVOLVER
             if ( mpi_subtask_command_flag && (aggregate_depth==0))
             { /* accumulate histogram data on master */
               REAL recvbuf[HISTBINS+1];
               MPI_Reduce(bins,recvbuf,HISTBINS+1,MPI_REAL,MPI_SUM,
                 MASTER_TASK, MPI_COMM_WORLD);
               if ( this_task == MASTER_TASK )
                 for ( k = 0, histo_count = 0 ; k <= HISTBINS ; k++ )
                 { bins[k] = recvbuf[k];
                   histo_count += (int)bins[k];
                 }
             }
#endif
             /* print histogram */
#ifdef MPI_EVOLVER
             if ( (this_task == MASTER_TASK) || !mpi_subtask_command_flag )
#endif
             {
             if ( histo_count == 0 )
             { 
               outstring("No qualifying values.\n");
               stacktop -= HISTBINS+1 + 3 + 3;
               break;
             }
             if ( one_bin_flag )
             { if (node->op1.aggrtype == HISTOGRAM_NODE )
                  sprintf(msg,"%10.5g - %10.5g     %d\n",
                  (DOUBLE)hi,(DOUBLE)hi,histo_count);
                else sprintf(msg,"%10.5g - %10.5g     %d\n",
                  (DOUBLE)exp(hi),(DOUBLE)exp(hi),histo_count);
               outstring(msg);
               stacktop -= HISTBINS+1 + 3 + 3;
               break;
             }
             if (node->op1.aggrtype == HISTOGRAM_NODE )
              for ( n = 0 ; n < HISTBINS ; n++ )
                { sprintf(msg,"%10.5g - %10.5g     %d\n",
                  (DOUBLE)(lo + n*binsize),(DOUBLE)(lo + (n+1)*binsize),
                     (int)bins[n]);
                  outstring(msg);
                }
              else /* log histogram */
                { int zeroes = (int)bins[0];
                  if ( zeroes )
                  { sprintf(msg,"          <= 0.0           %d\n",zeroes);
                    outstring(msg);
                  }
                  
                  for ( n = 1 ; n < HISTBINS ; n++ )
                  { sprintf(msg,"%10.5g - %10.5g     %d\n",
                     (DOUBLE)(exp(lo + (n-1)*binsize)),
                     (DOUBLE)(exp(lo + n*binsize)),
                     (int)bins[n]);
                    outstring(msg);
                  }
                }
              if ( bins[HISTBINS] > 0.0 ) 
              { sprintf(msg,"NaN          %d\n",(int)bins[HISTBINS]);
                outstring(msg);
              }
              }

              temp_free((char*)histo_data);
              stacktop -= HISTBINS+1 + 3 + 3;  /* erase locals */
              mpi_subtask_command_flag = 0;
              break;
            }

             case LIST_NODE:
             stacktop -= 3;  /* erase locals */
             break;

             case SET_COLOR_NODE: 
             case SET_FRONTCOLOR_NODE: 
             case SET_BACKCOLOR_NODE: 
             case SET_OPACITY_NODE:
             stacktop -= 3;  /* erase locals */
             update_display_flag = 1;
             break;

             case SET_FIXED_NODE: case SET_NO_REFINE_NODE: case SET_NO_DISPLAY_NODE:
             case SET_NO_TRANSFORM_NODE: case UNSET_NO_TRANSFORM_NODE:
             case SET_DENSITY_NODE: case UNSET_DENSITY_NODE:  case SET_BARE_NODE:
             case SET_CONSTRAINT_NODE: case UNSET_CONSTRAINT_NODE:
             case SET_CONSTRAINT_NAME_NODE: case UNSET_CONSTRAINT_NAME_NODE:
             case SET_VOLUME_NODE: case SET_PRESSURE_NODE: case SET_NONCONTENT_NODE:
             case UNSET_BOUNDARY_NODE: case UNSET_BOUNDARY_NAME_NODE:
             case UNSET_FIXED_NODE: case UNSET_PRESSURE_NODE: 
             case SET_NAMED_QUANTITY_NODE: case UNSET_NO_REFINE_NODE: case UNSET_BARE_NODE:
             case UNSET_NAMED_QUANTITY_NODE: case UNSET_NO_DISPLAY_NODE:
             case SET_METHOD_INSTANCE_NODE: case UNSET_METHOD_INSTANCE_NODE:
             case UNSET_TRIPLE_PT_NODE: case UNSET_TETRA_PT_NODE:
             case UNSET_NONCONTENT_NODE: case SET_BOUNDARY_NODE:
             case SET_HIT_PARTNER_NODE: case UNSET_HIT_PARTNER_NODE:
             case UNSET_CENTEROFMASS_NODE: case SET_CENTEROFMASS_NODE:
             case SET_NO_HESSIAN_NORMAL_NODE: case UNSET_NO_HESSIAN_NORMAL_NODE:
             stacktop -= 3;  /* erase locals */
             recalc_flag = 1;
             break;

             case SET_FRONTBODY_NODE:
             case SET_BACKBODY_NODE:
             case UNSET_FRONTBODY_NODE:
             case UNSET_BACKBODY_NODE:
             case UNSET_FACET_BODY_NODE:
                 /*make_bfacet_lists();  */ /* so lists legal */
                 stacktop -= 3;  /* erase locals */
                 recalc_flag = 1;
                 break;

              case REFINE_NODE:
                stacktop -= 3;  /* erase locals */
#ifdef MPI_EVOLVER
               if ( (this_task != MASTER_TASK) && mpi_subtask_command_flag )
                  mpi_task_edge_refine_wrapup();
#endif
                break;

              case DISSOLVE_NODE:
                stacktop -= 3;  /* erase locals */
                break;

              case DELETE_NODE:
                stacktop -= 3;  /* erase locals */
#ifdef MPI_EVOLVER
               if ( (this_task != MASTER_TASK) && mpi_subtask_command_flag )
                  mpi_task_delete_wrapup();
#endif
                break;

              case FIX_NODE:
                stacktop -= 3;  /* erase locals */
                break;

              case UNFIX_NODE:
                stacktop -= 3;  /* erase locals */
                break;

              default:
                stacktop -= 3;  /* erase locals */
                break; 
          } break; /* end AGGREGATE_END */

        /*********************/
        /* attribute setting */
        /* aggregate verbs    */
        /*********************/

      case SET_FIXED_NODE:
        set_attr(q_id,FIXED);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_FIXED_NODE:
        unset_attr(q_id,FIXED);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_BARE_NODE:
        set_attr(q_id,BARE_NAKED);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_BARE_NODE:
        unset_attr(q_id,BARE_NAKED);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_CENTEROFMASS_NODE:
        unset_attr(q_id,WANT_CENTEROFMASS);
        unset_attr(q_id,HAVE_CENTEROFMASS);

        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_CENTEROFMASS_NODE:
        set_attr(q_id,WANT_CENTEROFMASS);
        { REAL *cm = get_body_cm(q_id);  /* in unit cell coordinates */
          for ( i = 0 ; i < SDIM ; i++ )
            cm[i] = 0.5;
        }

        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_NO_DISPLAY_NODE:
        set_attr(q_id,NODISPLAY);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_NO_DISPLAY_NODE:
        unset_attr(q_id,NODISPLAY);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_NONCONTENT_NODE:
        if ( everything_quantities_flag )
        { sprintf(errmsg,"Changing NONCONTENT not implemented for everything_quantities mode yet.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2484,errmsg, RECOVERABLE);
        }
        set_attr(q_id,NONCONTENT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_NONCONTENT_NODE:
        if ( everything_quantities_flag )
        { sprintf(errmsg,"Changing NONCONTENT not implemented for everything_quantities mode yet.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2595,errmsg, RECOVERABLE);
        }
        unset_attr(q_id,NONCONTENT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_HIT_PARTNER_NODE:
        set_attr(q_id,HIT_PARTNER);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_HIT_PARTNER_NODE:
        unset_attr(q_id,HIT_PARTNER);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_NO_HESSIAN_NORMAL_NODE:
        set_attr(q_id,NO_HESSIAN_NORMAL_ATTR);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_NO_HESSIAN_NORMAL_NODE:
        unset_attr(q_id,NO_HESSIAN_NORMAL_ATTR);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_NO_REFINE_NODE:
        set_attr(q_id,NO_REFINE);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_NO_REFINE_NODE:
        unset_attr(q_id,NO_REFINE);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_NO_TRANSFORM_NODE:
        set_attr(q_id,NO_TRANSFORM);
        some_no_transforms_flag = 1; // have to disable transforms via graphics hardware
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_NO_TRANSFORM_NODE:
        unset_attr(q_id,NO_TRANSFORM);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_PRESSURE_NODE:
        unset_attr(q_id,PRESSURE);
        if ( everything_quantities_flag )
        { struct gen_quant *q = GEN_QUANT(get_body_volquant(q_id));
          q->modulus = 1;
          q->flags &= ~(Q_ENERGY|Q_FIXED|Q_CONSERVED);
          q->flags |= Q_INFO;
        }
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_CONSTRAINT_NODE:
      case UNSET_CONSTRAINT_NAME_NODE:
        if  (node->type == UNSET_CONSTRAINT_NODE) 
          k = (int)*(stacktop--);
        else
          k =  node->op3.connum;
        k &= CONMASK; /* so will ignore hit bit */
        if ( k < web.maxcon )
        switch ( id_type(q_id) )
        { case VERTEX:
             unset_v_constraint_map(q_id,k);
             break;
          case EDGE:
             unset_e_constraint_map(q_id,k);
             break;
          case FACET:
             unset_f_constraint_map(q_id,k);
             break;
          default: 
             sprintf(errmsg,"Bad element type for constraint.\n");
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1292,errmsg, RECOVERABLE);
        }
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_BOUNDARY_NODE:
        k =  (int)*(stacktop--);  /* boundary number */
        if ( get_attr(v_id) & CONSTRAINT )
        { sprintf(errmsg,"Cannot put %s %s on a boundary since it is already on a constraint.\n",
             typenames[id_type(q_id)],ELNAME(q_id));
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(5999,errmsg,RECOVERABLE);
        }
        bdry = web.boundaries+abs(k);
        if (  (abs(k) >= web.bdrymax) || !(bdry->attr&IN_USE) ) 
        { sprintf(errmsg,"Boundary %d is not valid.\n",k);
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2998,errmsg,RECOVERABLE);
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
          sprintf(errmsg,"Cannot put %s %s on boundary %d since it is already on another boundary.\n",
             typenames[id_type(q_id)],ELNAME(q_id),abs(k));
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2999,errmsg,RECOVERABLE);
        }
        set_attr(q_id,BOUNDARY);
        if ( k < 0 )
          set_attr(q_id,NEGBOUNDARY);
 
        switch(id_type(q_id))
          { case VERTEX:
            { REAL *x = get_coord(q_id);
              int n;
              set_boundary_num(q_id,abs(k));
              for ( n = 0 ; n < SDIM ; n++ )
              if ( bdry->coordf[n]->root != NULL )
              { PROF_EVAL_END(ex_profiling);
                PUSH_TRACE;
                x[n] = eval(bdry->coordf[n],get_param(q_id),q_id,NULL);
                POP_TRACE;
                PROF_EVAL_START(ex_profiling);
              }
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
             kb_error(3000,errmsg, RECOVERABLE);
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

      case UNSET_BOUNDARY_NODE:
      case UNSET_BOUNDARY_NAME_NODE:
        k = (node->type == UNSET_BOUNDARY_NODE) ? (int)*(stacktop--)
               : node->op3.bdrynum;
        if ( k >= web.bdrymax ) break;
        bdry = web.boundaries+k;
        if ( get_attr(q_id) & BOUNDARY )
        {
          switch ( id_type(q_id) )
          { case VERTEX:
             if ( get_boundary(q_id) == bdry )
              { set_boundary_num(q_id,0);
                unset_attr(q_id,BOUNDARY|HIT_PARTNER);
              }
             break;
            case EDGE:
             if ( get_edge_boundary(q_id) == bdry )
              { set_edge_boundary_num(q_id,0);
                unset_attr(q_id,BOUNDARY);
              }
             break;
            case FACET:
             if ( get_facet_boundary(q_id) == bdry )
              { set_facet_boundary_num(q_id,0);
                unset_attr(q_id,BOUNDARY);
              }
             break;
             default: 
             sprintf(errmsg,"Bad element type for boundary.\n");
             sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
             kb_error(1293,errmsg, RECOVERABLE);
           }
          if ( bdry->attr & CON_ENERGY )
             unapply_method(q_id,bdry->energy_method);
          if ( bdry->attr & CON_CONTENT )
           { if ( (web.representation == STRING) && (id_type(q_id) == VERTEX) )
              fixup_vertex_content_meths(q_id);
            else if ( (web.representation == SOAPFILM) && (id_type(q_id) == EDGE) )
              fixup_edge_content_meths(q_id);
          }
        }
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_VOLUME_NODE:
      case UNSET_TARGET_NODE:
        if ( get_attr(q_id) & FIXEDVOL)
         { unset_attr(q_id,FIXEDVOL);
           if (everything_quantities_flag )
           { struct gen_quant *q = GEN_QUANT(get_body_volquant(q_id));
             q->target = 0.0;
             if ( !(q->flags & Q_INFO) )
             { q->flags &= ~(Q_FIXED|Q_ENERGY|Q_CONSERVED);
                q->flags |= Q_INFO;
             }
             if ( web.pressure_flag )
             { q = GEN_QUANT(get_body_ambquant(q_id));
               q->flags &= ~(Q_FIXED|Q_ENERGY|Q_CONSERVED);
               q->flags |= Q_INFO;
             }
           }
         }
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_FACET_BODY_NODE:
        if ( id_type(q_id) != FACET  )
        { sprintf(errmsg,"Trying to unset body of non-facet.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                file_names[node->file_no],node->line_no);
          kb_error(1294,errmsg, RECOVERABLE);
        }
        set_facet_body(q_id,NULLID);
        set_facet_body(inverse_id(q_id),NULLID);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_FRONTBODY_NODE:
      case UNSET_BACKBODY_NODE:
       set_body(node->type==UNSET_FRONTBODY_NODE?q_id:inverse_id(q_id),NULLID);
       recalc_flag = 1;
       node += node->op1.skipsize - 1;  /* back to start of loop */
       break;

      case UNSET_DENSITY_NODE:
        unset_attr(q_id,DENSITY);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;
 
      case SET_TRIPLE_PT_NODE:
        set_attr(q_id,TRIPLE_PT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_TRIPLE_PT_NODE:
        unset_attr(q_id,TRIPLE_PT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_TETRA_PT_NODE:
        set_attr(q_id,TETRA_PT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_TETRA_PT_NODE:
        unset_attr(q_id,TETRA_PT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_AXIAL_POINT_NODE:
        set_attr(q_id,AXIAL_POINT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case UNSET_AXIAL_POINT_NODE:
        unset_attr(q_id,AXIAL_POINT);
        /* go back to next element generator */
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_DENSITY_NODE:
      { REAL density = *(stacktop--);
         if ( density != 1.0 ) set_attr(q_id,DENSITY);
         switch ( node->op2.eltype )
            { case EDGE: set_edge_density(q_id,density); break;
              case FACET: set_facet_density(q_id,density); break;
              case BODY: set_body_density(q_id,density); break;
            }
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
        }
      break;

      case SET_VOLUME_NODE: 
      { REAL v = *(stacktop--);
        if ( get_attr(q_id) & PRESSURE )
        { sprintf(errmsg,"Must unset body pressure before fixing target.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2071,errmsg, RECOVERABLE);
        }
        set_attr(q_id,FIXEDVOL);
        set_body_fixvol(q_id,v); 
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
        break;
      }

      case SET_PRESSURE_NODE: 
      { REAL p = *(stacktop--);
        if ( get_attr(q_id) & FIXEDVOL )
        { sprintf(errmsg,"Must unset body target before fixing pressure.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2521,errmsg, RECOVERABLE);
        }
        set_attr(q_id,PRESSURE);
        set_body_pressure(q_id,p); 
        if ( everything_quantities_flag )
        { struct gen_quant *q = GEN_QUANT(get_body_volquant(q_id));
          q->modulus = -p;
          if ( !(q->flags & Q_ENERGY) )
          { q->flags &= ~(Q_INFO|Q_FIXED|Q_CONSERVED);
            q->flags |= Q_ENERGY;
          }
        }
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
        break;
      }

      case SET_OPACITY_NODE:
        facet_alpha = *(stacktop)--;
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_CONSTRAINT_NODE: 
      case SET_CONSTRAINT_NAME_NODE:
       { int con = (node->type==SET_CONSTRAINT_NODE) ? (int)*(stacktop--)
                : node->op3.connum;
         if ( (con<0) || (con>=web.maxcon) || !(get_constraint(con)->attr & IN_USE))
         { sprintf(errmsg,"Illegal constraint number: %d.\n",con);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(1295,errmsg,RECOVERABLE);
         }
         if ( get_attr(q_id) & BOUNDARY )
         { sprintf(errmsg,
             "Cannot set %s %s on a constraint since it is on a boundary.\n",
                typenames[node->op2.eltype],ELNAME(q_id));
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no); 
           kb_error(4002,errmsg,RECOVERABLE);
         }
         switch ( node->op2.eltype )
            { case VERTEX: set_v_constraint_map(q_id,con);
                           project_v_constr(q_id,ACTUAL_MOVE,RESET_ONESIDEDNESS);
                           break;
              case EDGE: set_e_constraint_map(q_id,con); break;
              case FACET: set_f_constraint_map(q_id,con); break;
            }
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
       }
       break;

      case SET_NAMED_QUANTITY_NODE:
       { int qnum = (int)*(stacktop--);
         apply_quantity(q_id,qnum);
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
        }
        break;

      case UNSET_NAMED_QUANTITY_NODE:
       { int qnum = (int)*(stacktop--);
         unapply_quantity(q_id,qnum);
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
       }
       break;

      case SET_METHOD_INSTANCE_NODE:
       { int qnum = (int)*(stacktop--);
         apply_method_num(q_id,qnum);
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
       }
       break;

      case UNSET_METHOD_INSTANCE_NODE:
       { int qnum = (int)*(stacktop--);
         unapply_method(q_id,qnum);
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
       }
       break;

      case SET_COLOR_NODE:
       { int color = (int)*(stacktop--);
         switch ( node->op2.eltype )
            { case EDGE: set_edge_color(q_id,color); break;
              case FACET: set_facet_color(q_id,color); break;
            }
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
         update_display_flag = 1;
       }
       break;

      case SET_FRONTCOLOR_NODE:
      { int color = (int)*(stacktop--);
         set_facet_frontcolor(q_id,color);
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
         update_display_flag = 1;
        }
        break;

      case SET_BACKCOLOR_NODE:
      { int color = (int)*(stacktop--);
         set_facet_backcolor(q_id,color);
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
         update_display_flag = 1;
        }
        break;

      case SET_COORD_NODE:
      case SET_COORD_1_NODE:  // Listing all cases here for node_extract
      case SET_COORD_2_NODE:
      case SET_COORD_3_NODE:
      case SET_COORD_4_NODE:
      case SET_COORD_5_NODE:
      case SET_COORD_6_NODE:
      case SET_COORD_7_NODE:
      case SET_COORD_8_NODE:
      case SET_COORD_9_NODE:
      case SET_COORD_10_NODE:
         get_coord(q_id)[node->op2.coordnum] = *(stacktop--);
         if ( node->op2.coordnum <= SDIM ) recalc_flag = 1;
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_PARAM_NODE:
      case SET_PARAM_1_NODE:
      case SET_PARAM_2_NODE:
      case SET_PARAM_3_NODE:
      case SET_PARAM_4_NODE:
      case SET_PARAM_5_NODE:
      case SET_PARAM_6_NODE:
      case SET_PARAM_7_NODE:
      case SET_PARAM_8_NODE:
       if ( get_vattr(q_id) & BOUNDARY )
          { struct boundary *boundary = get_boundary(q_id);
             REAL *param = get_param(q_id);
             REAL *xx = get_coord(q_id);
             if ( node->op2.coordnum > boundary->pcount )
             { sprintf(errmsg,"Parameter number is %d; maximum is %d.\n",
                 node->op2.coordnum,boundary->pcount); 
               sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
               kb_error(1296,errmsg,RECOVERABLE);
             }
             param[node->op2.coordnum] = *(stacktop--);
             PROF_EVAL_END(ex_profiling);
             PUSH_TRACE;
             for ( k = 0 ; k < SDIM ; k++ )
               xx[k] = eval(boundary->coordf[k],param,q_id,NULL);
             POP_TRACE;
             PROF_EVAL_START(ex_profiling);
             recalc_flag = 1;
          }
         else
          { /* probably just handy for storage */
             REAL *param = get_param(q_id);
             param[node->op2.coordnum] = *(stacktop--);
          }
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case SET_EXTRA_ATTR_NODE:
      case SET_FRONTBODY_NODE:
      case SET_BACKBODY_NODE:
      case SET_ORIGINAL_NODE:
      case SET_ORIENTATION_NODE:
       sprintf(errmsg,"Internal error; SET node %d reached unexpectedly.\n",node->type);
       sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
       kb_error(5387,errmsg,RECOVERABLE);
       break;

        /************************/
        /* numerical aggregates */
        /************************/


      case MAX_NODE: 
       { REAL *aggrptr;

         val = *(stacktop--);
         aggrptr = (REAL *)(stacktop - 3);
         if ( val > *aggrptr ) *aggrptr = val;
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
       }
       break;

      case MIN_NODE: 
       {
         REAL *aggrptr;

         val = *(stacktop--);
         aggrptr = (REAL *)(stacktop - 3);
         if ( val < *aggrptr ) *aggrptr = val;
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
       }
       break;

      case SUM_NODE: 
         stacktop--;
         stacktop[-3] += stacktop[1];
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case AVG_NODE: 
         stacktop--;
         stacktop[-3] += stacktop[1]; /* sum */
         stacktop[-4] += 1.0; /* count */
         /* go back to next element generator */
         node += node->op1.skipsize - 1;  /* back to start of loop */
         break;

      case COUNT_NODE: 
          stacktop--;
          stacktop[-3] += 1.0;
            /* go back to next element generator */
              node += node->op1.skipsize - 1;  /* back to start of loop */
          break;

      case HISTOGRAM_NODE:   
      case LOGHISTOGRAM_NODE:
          if ( histo_count >= histo_max )
          { histo_data = (REAL*)temp_realloc((char*)histo_data,
                  2*histo_max*sizeof(REAL));
            histo_max *= 2;
          }
          histo_data[histo_count++] = *(stacktop--);
          node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case FOREACH_NODE:
      case SET_ATTRIBUTE_LOOP_NODE:
        node += node->op1.skipsize - 1;  /* back to start of loop */
        break;

      case ARRAY_HEAD_NODE: break;  /* let indices accumulate */

      case ARRAYASSIGN_NODE: 
      { struct array *a;
        REAL value=0.0,rhs;
        int i,offset;
        void *lvalue;
        struct global *g = globals(node->op2.name_id);
 
        a = get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);

        rhs = *(stacktop--);
        for ( i = 0 ; i < a->dim ; i++ )
        { int k = (int)stacktop[i+1-a->dim];
          if ( k < 1 )
          { sprintf(errmsg,
             "Array index %d of array %s is %d. Indexes start at 1.\n",
               i+1,g->name,k);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(3011,errmsg,RECOVERABLE);
          }
          if ( k > a->sizes[i] && (!(node->flags & IS_VIRTUAL_ATTR) ||(k > SDIM)) )
          { sprintf(errmsg,"Array index %d of array %s is %d; exceeds bound of %d.\n",
               i+1,g->name,k,a->sizes[i]);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(3012,errmsg,RECOVERABLE);
          }
        }
        for ( i = 1, offset = (int)stacktop[1-a->dim]-1 ; i < a->dim ; i++ )
        { offset *= a->sizes[i];
          offset += (int)stacktop[i+1-a->dim]-1;  /* 1-based indexing */
        }
        stacktop -= a->dim;
        lvalue = ((char *)a) + a->datastart + offset*a->itemsize;
        switch ( a->datatype )
        { case REAL_TYPE: value = *(REAL*)(lvalue);  break;
          case INTEGER_TYPE: value = *(int*)(lvalue); break;
          case UINT_TYPE: value = *(unsigned int*)(lvalue); break;
          case SHORT_TYPE: value = *(short int*)(lvalue); break;
          case USHORT_TYPE: value = *(unsigned short int*)(lvalue); break;
          case LONG_TYPE: value = *(long int*)(lvalue); break;
          case ULONG_TYPE: value = *(unsigned long int*)(lvalue); break;
          case CHAR_TYPE: value = *(char*)(lvalue); break;
          case UCHAR_TYPE: value = *(unsigned char*)(lvalue); break;
          case PTR_TYPE: value = (unsigned long int)*(char**)(lvalue); break;
          case VERTEX_TYPE:
          case EDGE_TYPE:
          case FACET_TYPE:
          case BODY_TYPE:
          case FACETEDGE_TYPE:
          case ELEMENTID_TYPE:  break;
          default: value = *(int*)(lvalue); break;
        
        }
        switch ( node->op1.assigntype )
        {
          case ASSIGN_OP:  value = rhs;  break;
          case PLUSASSIGN_OP: value += rhs; break;
          case SUBASSIGN_OP: value -= rhs;   break;
          case MULTASSIGN_OP: value *= rhs; break;
          case DIVASSIGN_OP:
            if( rhs == 0.0 )
            { sprintf(errmsg,"Division by zero.\n");
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(3015,errmsg,RECOVERABLE);
            }
            value /= rhs;  break;
        } 
        switch ( a->datatype )
        { case REAL_TYPE: *(REAL*)(lvalue) = value;    break;
          case INTEGER_TYPE: *(int*)(lvalue) = (int)value; break;
          case UINT_TYPE: *(unsigned int*)(lvalue) = (unsigned int)value; break;
          case CHAR_TYPE: *(char*)(lvalue) = (char)value; break;
          case UCHAR_TYPE: *(unsigned char*)(lvalue) = (unsigned char)value; break;
          case SHORT_TYPE: *(short*)(lvalue) = (short)value; break;
          case USHORT_TYPE: *(unsigned short*)(lvalue) = (unsigned short)value; break;
          case LONG_TYPE: *(long*)(lvalue) = (long)value; break;
          case ULONG_TYPE: *(unsigned long*)(lvalue) = (unsigned long)value; break;
          case VERTEX_TYPE:
          case EDGE_TYPE:
          case FACET_TYPE:
          case BODY_TYPE:
          case FACETEDGE_TYPE:
          case ELEMENTID_TYPE:  break;
          default: *(int*)(lvalue) = (int)value; break;
        
        }
        if ( g->flags & RECALC_PARAMETER )
            recalc_flag = 1;
        if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
            dont_resize_flag = 1;

      }
      break;


      case ARRAYEVAL_NODE:
      { 
        struct global *g = globals(node->op2.name_id);
        struct array *a;
        REAL value=0.0;
        int i,offset;
        void *lvalue;
 
        a = get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);

        for ( i = 0 ; i < a->dim ; i++ )
        { int k = (int)stacktop[i+1-a->dim];
          if ( k < 1 )
          { sprintf(errmsg,
              "Array index %d of array %s is %d. Indexes start at 1.\n",
               i+1,g->name,k);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2047,errmsg,RECOVERABLE);
          }
          if ( k > a->sizes[i] )
          { sprintf(errmsg,
             "Array index %d of array %s is %d; exceeds bound of %d.\n",
               i+1,g->name,k,a->sizes[i]);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(2048,errmsg,RECOVERABLE);
          }
        }
        for ( i = 1, offset = (int)stacktop[1-a->dim]-1 ; i < a->dim ; i++ )
        { offset *= a->sizes[i];
          offset += (int)stacktop[i+1-a->dim]-1;  /* 1-based indexing */
        }
        stacktop -= a->dim;
        lvalue = ((char *)a) + a->datastart + offset*a->itemsize;
        switch ( a->datatype )
        { case REAL_TYPE: value = *(REAL*)(lvalue);    break;
          case INTEGER_TYPE: value = *(int*)(lvalue);    break;
          case UINT_TYPE: value = *(unsigned int*)(lvalue);    break;
          case CHAR_TYPE: value = *(char*)(lvalue);    break;
          case UCHAR_TYPE: value = *(unsigned char*)(lvalue);    break;
          case SHORT_TYPE: value = *(short int*)(lvalue);    break;
          case USHORT_TYPE: value = *(unsigned short int*)(lvalue);    break;
          case LONG_TYPE: value = *(long*)(lvalue);    break;
          case ULONG_TYPE: value = *(unsigned long*)(lvalue);    break;
          case PTR_TYPE: value = (unsigned long)*(char**)(lvalue);    break;
          case VERTEX_TYPE:
          case EDGE_TYPE:
          case FACET_TYPE:
          case BODY_TYPE:
          case FACETEDGE_TYPE:
          case ELEMENTID_TYPE: value = (int)*(element_id*)(lvalue); break;
          default: value = *(int*)(lvalue); break;
        }
        *(++stacktop) = value;
      }
      break;

    /* whole array syntax */
    
      case ARRAYIDENT_NODE: /* push datastart for array */
        { struct global *glvalue = globals(node->op2.name_id);
          struct array *alvalue;
          alvalue = get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          if ( glvalue->flags & FIXED_SIZE_ARRAY )
            *(REAL**)(++stacktop) = 
                newstack + localstack + glvalue->attr.arrayptr->datastart;
          else 
            *(char**)(++stacktop) = (char*)alvalue + alvalue->datastart;
          break;
        }

        case FIXED_ARRAY_RVAL_NODE: /* push fixed index entry of fixed-size local */
        { 
          struct array *alvalue;
          alvalue = get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          *(++stacktop) = 
                *(newstack + localstack + alvalue->datastart + node->op3.offset);
          node += node->op4.skip;
          break;
        }

      case ATTRIB_LVALUE_NODE:  /* push datastart for attribute array */
        { element_id id;
          n = node->op2.name_id & GLOBMASK; /* attribute number */
          if ( node->op1.localnum )
            id = *(element_id*)get_localp(node->op1.localnum);
          else id = q_id;
          *(char**)(++stacktop) = (char*)get_extra(id,n);
        }
        break;

      case ARRAY_VERTEX_NORMAL_NODE:
      case ARRAY_EDGE_VECTOR_NODE:
      case ARRAY_FACET_NORMAL_NODE:
        { element_id id;
          REAL *datastart =  get_localp(node->op3.localnum);
          *(REAL**)(++stacktop) = datastart;
          if ( node->flags & IS_RVALUE )
          { if ( node->op1.localnum )
              id = *(element_id*)get_localp(node->op1.localnum);
            else id = q_id;
            switch ( node->type )
            { case ARRAY_VERTEX_NORMAL_NODE:
               { MAT2D(normal,MAXCOORD,MAXCOORD);
                 REAL mag;
                 int normcount;

                 normcount = new_calc_vertex_normal(id,normal);
           //      project_vertex_normals(id,normal,normcount);
                 mag = sqrt(SDIM_dot(normal[0],normal[0]));
                 if ( mag == 0.0 ) mag = 1;
                 for ( i = 0 ; i < SDIM ; i++ )
                    datastart[i] = normal[0][i]/mag;
              
                 break;
                }
              case ARRAY_EDGE_VECTOR_NODE:
                 get_edge_side(id,datastart);
                 break;
              case ARRAY_FACET_NORMAL_NODE:
                 get_facet_normal(id,datastart);
                 break;
            }
          }
        }
        break;

      case ARRAY_VERTEX_CONSTRAINT_NORMAL_NODE:
        { vertex_id v_id;
          REAL *datastart =  get_localp(node->op3.localnum);
          REAL fval; // dummy for eval_all()
          REAL mag;
          int connum = (int)(*stacktop--);
          struct constraint *con;

          *(REAL**)(++stacktop) = datastart;
          if ( node->op1.localnum )
              v_id = *(element_id*)get_localp(node->op1.localnum);
          else v_id = q_id;
          if ( connum < 0 || connum >= web.maxcon )
          { sprintf(errmsg,"constraint_normal: Constraint %d does not exist.\n",connum);
            kb_error(4322,errmsg,RECOVERABLE);
          }
          con = get_constraint(connum);
          if ( !(con->attr & IN_USE) )
          { sprintf(errmsg,"constraint_normal: Constraint %d does not exist.\n",connum);
            kb_error(4323,errmsg,RECOVERABLE);
          }
          eval_all(con->formula,get_coord(v_id),SDIM,&fval,datastart,v_id);
          // normalize
          mag = SDIM_dot(datastart,datastart);
          if ( mag > 0 )
          { mag = sqrt(mag);
            for ( i = 0 ; i < SDIM ; i++ )
                datastart[i] /= mag;
          }
        }
        break;

      case ARRAY_LVALUE_INDEXED_NODE:
      case ARRAY_RVALUE_INDEXED_NODE:
      { 
        struct array *a;
        int offset;

        /* Adjust datastart on stack by indexes */
        if ( node->op5.indexcount == 0 )
          break;  // no indices, so no adjustment needed
          
        a = get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);

        for ( i = 0 ; i < node->op5.indexcount ; i++ )
        { int k = (int)stacktop[i+1-node->op5.indexcount];
          if ( k < 1 )
          { sprintf(errmsg,
             "Array index %d of array %s is %d. Indexes start at 1.\n",
               i+1,get_name_name(node->op2.name_id,localbase),k);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(3022,errmsg,RECOVERABLE);
          }
          if ( k > a->sizes[i] && (!(node->flags & IS_VIRTUAL_ATTR) ||(k > SDIM)) )
          {sprintf(errmsg,"Array index %d of array %s is %d; exceeds bound of %d.\n",
               i+1,get_name_name(node->op2.name_id,localbase),k,a->sizes[i]);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(3023,errmsg,RECOVERABLE);
          }
        }

        for ( i = 1, offset = (int)stacktop[1-node->op5.indexcount]-1 ; i < node->op5.indexcount ; i++ )
        { offset *= a->sizes[i];
          offset += (int)stacktop[i+1-node->op5.indexcount]-1;  /* 1-based indexing */
        }
        for ( ; i < a->dim ; i++ )
          offset *= a->sizes[i];
        stacktop -= node->op5.indexcount;
        *(char**)(stacktop) += offset*a->itemsize;

        // fix up *a for use at actual assignment
        /*
        a->datacount = 1;
        for ( i = node->op5.indexcount ; i < a->dim ; i++ )
        { a->datacount *= a->sizes[i];
          a->sizes[i-node->op5.indexcount] = a->sizes[i];
        }
        a->dim -= node->op5.indexcount;
        for ( i = 0 ; i < node->op5.indexcount ; i++ )
          a->sizes[a->dim+i] = 0;
          */

        break;
      }

      case ARRAY_RVALUE_NODE:
        break;
        
      case DOT_NODE:  /* dot product */
        { struct array *a,*b;
          int name1 = node->op2.name_id;
          int name2 = node->op3.name_id;
          REAL *datastart1,*datastart2;
          REAL sum;
          int counta,countb,count;
          if ( node[node->left].flags & IS_VIRTUAL_ATTR )
            counta = SDIM;
          else
          { a = get_name_arrayptr(name1,newstack+localstack,localbase);
            counta = a->sizes[a->dim-1];
          }
          if ( node[node->right].flags & IS_VIRTUAL_ATTR )
            countb = SDIM;
          else
          { b = get_name_arrayptr(name2,newstack+localstack,localbase);
            countb = b->sizes[b->dim-1];
          }
          count = (counta < countb) ? counta : countb;
          datastart1 = *(REAL**)(stacktop--);
          datastart2 = *(REAL**)(stacktop--);
          for ( sum = 0.0, i = 0 ; i < count ; i++ )
            sum += datastart1[i]*datastart2[i];
          *(++stacktop) = sum;
          break;
        }
 
 
      case ARRAY_EVAL_NODE:  /* rexpr: arraylvalue indexset */
      { /* use info on stack to push value of array element.
             stack: datastart index-values -> rexpr */
        REAL value=0.0;
        void *lvalue;

        lvalue = *(char**)(stacktop--);
        
        switch ( node[node->left].datatype )
        { case STRING_TYPE: *(char**)(++stacktop) = *(char**)lvalue; goto adfadf;  // just keep pointer to string
          case REAL_TYPE: value = *(REAL*)(lvalue);  break;
          case INTEGER_TYPE: value = *(int*)(lvalue); break;
          case UINT_TYPE: value = *(unsigned int*)(lvalue); break;
          case SHORT_TYPE: value = *(short int*)(lvalue); break;
          case USHORT_TYPE: value = *(unsigned short int*)(lvalue); break;
          case LONG_TYPE: value = *(long int*)(lvalue); break;
          case ULONG_TYPE: value = *(unsigned long int*)(lvalue); break;
          case CHAR_TYPE: value = *(char*)(lvalue); break;
          case UCHAR_TYPE: value = *(unsigned char*)(lvalue); break;
          case PTR_TYPE: value = (unsigned long int)*(char**)(lvalue); break;
          case VERTEX_TYPE:
          case EDGE_TYPE:
          case FACET_TYPE:
          case BODY_TYPE:
          case FACETEDGE_TYPE:
          case ELEMENTID_TYPE: value = ordinal(*(element_id*)lvalue)+1; break;
          default: value = *(int*)(lvalue); break;
        
        }
        *(++stacktop) = value;
adfadf:
        break;
      }
        
      case ARRAY_ASSIGNOP_ARRAY_NODE:
        { 
          struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          struct array *arvalue = get_name_arrayptr(node->op3.name_id,newstack + localstack,localbase);
          int inx[100]; /* keep track of indices */
          char *p,*q;
          int j,k;
          char *pspots[MAXARRAYDIMS],*qspots[MAXARRAYDIMS];
          int pstride[MAXARRAYDIMS],qstride[MAXARRAYDIMS];
          int minsize[MAXARRAYDIMS],lastsize;
          int dim;
          int lcount = node[node->left].op5.indexcount;
          int rcount = node[node->right].op5.indexcount;
          int pint = (alvalue->datatype == INTEGER_TYPE);
          int qint = (arvalue->datatype == INTEGER_TYPE);
          
          /* Was checked in parser that the arrays have the same number
             of indices, but might be different sizes. 
           */
 
          /* pop datastarts off stack */
          q = *(char**)(stacktop--);
          p = *(char**)(stacktop--);

          dim = arvalue->dim - rcount;
          for ( i = 0 ; i < dim ; i++ )
          { inx[i] = 0;
            minsize[i] = arvalue->sizes[rcount+i] < alvalue->sizes[lcount+i] ?
                          arvalue->sizes[rcount+i] : alvalue->sizes[lcount+i] ;
            pspots[i] = p;
            qspots[i] = q;
          }
          if ( dim >= 2 )
          {
            pstride[dim-2] = alvalue->sizes[alvalue->dim-1]*alvalue->itemsize;
            qstride[dim-2] = arvalue->sizes[arvalue->dim-1]*arvalue->itemsize;
            for ( i = alvalue->dim - 3 ; i >= lcount ; i-- )
              pstride[i] = pstride[i+1]*alvalue->sizes[i+1];
            for ( i = arvalue->dim - 3 ; i >= rcount ; i-- )
              qstride[i] = qstride[i+1]*arvalue->sizes[i+1];
          }
          lastsize = minsize[dim-1];

          do
          { /* do a row */
           
                int *pi = (int*)p;
                REAL *pr = (REAL*)p;
                int *qi = (int*)q;
                REAL *qr = (REAL*)q;
                 switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) = qint ? *(qi++) : (int)*(qr++);
                       else
                         *(pr++) = qint ? *(qi++) : *(qr++);
                     break;
                  case PLUSASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) += qint ? *(qi++) : (int)*(qr++);
                       else
                         *(pr++) += qint ? *(qi++) : *(qr++);
                     break;
                  case SUBASSIGN_OP:
                    for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) -= qint ? *(qi++) : (int)*(qr++);
                       else
                         *(pr++) -= qint ? *(qi++) : *(qr++);
                     break;                       
                  case MULTASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) *= qint ? *(qi++) : (int)*(qr++);
                       else
                         *(pr++) *= qint ? *(qi++) : *(qr++);
                     break;
                  case DIVASSIGN_OP:
                    for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                       { *pi = (int)(*pi /(qint ? (REAL)*(qi++) : *(qr++)));
                          pi++;
                       }
                       else
                         *(pr++) /= qint ? (REAL)*(qi++) : *(qr++);
                     break;
                }
             

            /* increment pointers */
            for ( j = dim-2 ; j >= 0 ; j-- )
            { pspots[j] += pstride[j];
              qspots[j] += qstride[j];
              if ( ++inx[j] < minsize[j] )
              { p = pspots[j];
                q = qspots[j];
                for ( k = j+1 ; k < dim-1 ; k++ )
                { pspots[k] = p;
                  qspots[k] = q;
                }
                break;
              }
              inx[j] = 0 ;
            }  
           } while ( j >= 0 ); 
              
          if ( node->flags & RECALC_FLAG )
            recalc_flag = 1;
          if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
            dont_resize_flag = 1;

          break;
        } /* end ARRAY_ASSIGNOP_ARRAY_ */

      case ARRAY_ASSIGNOP_SCALAR_NODE: 
        { 
          struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          REAL scalar = *(stacktop--);
          char *p;
          int lastsize = alvalue->datacount;

          if ( node[node->left].type == ARRAY_LVALUE_INDEXED_NODE )
          { int k;
            lastsize = 1;
            for ( k = node[node->left].op5.indexcount ; k < alvalue->dim ; k++ )
              lastsize *= alvalue->sizes[k];
          }
          p = *(char**)(stacktop--);
          switch ( alvalue->datatype )
          { case REAL_TYPE:
            { REAL *pp = (REAL*)p;

                switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) = scalar;
                     break;
                  case PLUSASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) += scalar;
                     break;
                  case SUBASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) -= scalar;
                     break;                        
                  case MULTASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) *= scalar;
                     break;
                  case DIVASSIGN_OP:
                     if ( scalar == 0.0 )
                     {
                       sprintf(errmsg,"Division by zero.\n");
                       sprintf(errmsg+strlen(errmsg),
                          "(source file %s, line %d)\n",
                             file_names[node->file_no],node->line_no);
                       kb_error(4676,errmsg,RECOVERABLE);
                     }
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) /= scalar;
                     break;
                }
                break; /* end REAL_TYPE */
              }

              case INTEGER_TYPE:
              { int intscalar = (int)floor(scalar);
                int *pp = (int*)p;

                switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) = intscalar;
                     break;
                  case PLUSASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) += intscalar;
                     break;
                  case SUBASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) -= intscalar;
                     break;                        
                  case MULTASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                     { *pp = (int)(*(int*)pp * scalar);
                       pp++;
                     }
                     break;
                  case DIVASSIGN_OP:
                     if ( scalar == 0.0 )
                     { sprintf(errmsg,"Division by zero.\n");
                       sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                       kb_error(4679,errmsg,RECOVERABLE);
                     }
                     for ( i = 0 ; i < lastsize ; i++ ) 
                     { *pp = (int)((*(int*)pp) / scalar);
                        pp++;
                     }
                     break;
                }
              }
              break; /* end INTEGER_TYPE */
            }         

           if ( node->flags & RECALC_FLAG )
             recalc_flag = 1;
           if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
             dont_resize_flag = 1;

         } /* end ARRAY_ASSIGNOP_SCALAR */
         break;

      case ARRAY_ASSIGNOP_STRING_NODE: 
        { struct global *g = globals(node->op1.name_id);
          struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          char *s = *(char**)(stacktop--);
          char **p;
          char **pp;
          int lastsize = alvalue->datacount;

          if ( node[node->left].type == ARRAY_LVALUE_INDEXED_NODE )
          { int k;
            lastsize = 1;
            for ( k = node[node->left].op5.indexcount ; k < alvalue->dim ; k++ )
              lastsize *= alvalue->sizes[k];
          }
          p = *(char***)(stacktop--);
          pp = p;

          for ( i = 0 ; i < lastsize ; i++,pp++ ) 
          { if ( g->flags & GLOB_LOCALVAR )
              *(char**)pp = temp_realloc(*(char**)pp,(s?strlen(s):0)+1);
            else
              *(char**)pp = kb_realloc(*(char**)pp,(s?strlen(s):0)+1);
            strcpy(*(char**)pp,s);
          }
               
          if ( node->left && (node[node->left].flags & DEALLOCATE_POINTER) )
            myfree(s);  // for sprintf source of right side

           if ( node->flags & RECALC_FLAG )
             recalc_flag = 1;
           if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
             dont_resize_flag = 1;

         } /* end ARRAY_ASSIGNOP_STRING_NODE */
         break;

        case ARRAYLIST_NODE:
          break;

        case ARRAYEXPR_NODE:
          // push marker and dimension level
          *(++stacktop) = ARRAYINIT_MARKER;
          *(int *)(++stacktop) = node->op4.arraydim;
          break;

        case ARRAYEXPR_ASSIGN_NODE: // initializer syntax on the right 
        { 
          struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          REAL *spot = stacktop - node->op3.argcount + 1;  // start of data
          void *pbase = *(void**)(stacktop - node->op3.argcount);  // destination data pointer
          int lastsize = alvalue->sizes[alvalue->dim-1];  // row size
          int lowdim = node->op5.indexcount;
          int poffset; // number of elements linear offset in destination
          void *prevpp = pbase;  // for filling with zeros since last spot
          int pspots[MAXARRAYDIMS];  // indices of current slice

       /* NEED INDEXCOUNT IN NODE.OP5, OR DIM OF RHS */

          memset(pspots,0,sizeof(pspots));

          for(;;)
          {  int d;
            
            // first, calculate destination spot
            int size;  // of slice so far, in items
            for ( i = alvalue->dim-1, size = 1, poffset = 0 ; i >= lowdim ; i-- )
            { poffset += size*pspots[i];
              size *= alvalue->sizes[i];
            }

            switch ( alvalue->datatype )
            { case REAL_TYPE:
              { REAL *pp = (REAL*)pbase + poffset;

                switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                   { REAL *ppp = (REAL*)prevpp; 
                     for ( ; ppp != pp ; ppp++ )  // fill with 0 since last spot
                       *ppp = 0;
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) = *(spot++);
                     for ( ; i < lastsize ; i++ )
                       *(pp++) = 0;
                     prevpp = pp;
                     break;
                   }
                  case PLUSASSIGN_OP:
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) += *(spot++);
                     break;
                  case SUBASSIGN_OP:
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) -= *(spot++);
                     break;                        
                  case MULTASSIGN_OP:
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) *= *(spot++);
                     break;
                  case DIVASSIGN_OP:
                     if ( *spot == 0.0 )
                     {
                       sprintf(errmsg,"Division by zero.\n");
                       sprintf(errmsg+strlen(errmsg),
                          "(source file %s, line %d)\n",
                             file_names[node->file_no],node->line_no);
                       kb_error(4677,errmsg,RECOVERABLE);
                     }
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) /= *(spot++);
                     break;

                 }
                break; /* end REAL_TYPE */
              }

              case INTEGER_TYPE:
              { 
                int *pp = (int*)pbase + poffset;

                switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                   { int *ppp = (int*)prevpp; 
                     for ( ; ppp != pp ; ppp++ )  // fill with 0 since last spot
                       *ppp = 0;
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) = (int)floor(*(spot++));
                     for ( ; i < lastsize ; i++ )
                       *(pp++) = 0;
                     prevpp = pp;
                     break;
                   }
                  case PLUSASSIGN_OP:
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) += (int)floor(*(spot++));
                     break;
                  case SUBASSIGN_OP:
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) -= (int)floor(*(spot++));
                     break;                        
                  case MULTASSIGN_OP:
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) *= (int)floor(*(spot++));
                     break;
                  case DIVASSIGN_OP:
                     if ( *spot == 0.0 )
                     { sprintf(errmsg,"Division by zero.\n");
                       sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                       kb_error(4678,errmsg,RECOVERABLE);
                     }
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                       *(pp++) /= (int)floor(*(spot++));
                     break;
                }
              } 
              break; /* end INTEGER_TYPE */

              case STRING_TYPE:
              { char **pp = (char**)pbase + poffset;
                

                switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                   { char **ppp = (char**)prevpp; 
                     for ( ; ppp != pp ; ppp++ )  // fill with 0 since last spot
                       *ppp = 0;
                     for ( i = 0; (i < lastsize) && (*spot != ARRAYINIT_MARKER); i++ )
                     { char **cspot = (char**)spot;
                       char *s = kb_realloc(*pp,strlen(*cspot)+2);
                       strcpy(s,*cspot);
                       *(pp++) = s;
                       spot++;
                     }
                     for ( ; i < lastsize ; i++ )
                       *(pp++) = 0;
                     prevpp = pp;
                     break;
                   }
                  default: 
                    kb_error(2665,"Invalid string assignment operator; only plain assignment allowed.\n",
                      RECOVERABLE);
                }
              }
              break; /* end STRING_TYPE */

              kb_error(5567,"Illegal array initializer type.\n",RECOVERABLE);
            }         

            // if too many initializers, skip rest
            while ( *spot != ARRAYINIT_MARKER )
              spot++;

            // update destination pointer
            for(;;)
            { spot++;
              d = *(int*)(spot++) + lowdim;
              if ( d == alvalue->dim ) break;
              // update pspots
              pspots[alvalue->dim-d-1]++;
              if ( pspots[alvalue->dim-d-1] == alvalue->sizes[alvalue->dim-d-1] )
              { // find proper marker in initializer
                d++;
                while ( !((*spot == ARRAYINIT_MARKER) && (*(int*)(spot+1) == d-lowdim)) )
                  spot++;
              }
              else break;
            }
            if ( d == alvalue->dim ) break;  // all done

            // reset inner slice indices
            for ( i = alvalue->dim-d ; i < alvalue->dim ; i++ )
               pspots[i] = 0;                     

          } // end for

          // pop stuff
          stacktop -= node->op3.argcount + 1;  // +1 for destination pointer

          if ( node->flags & RECALC_FLAG )
            recalc_flag = 1;
          if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
             dont_resize_flag = 1;

        } /* end ARRAYEXPR_ASSIGN */
        break;

      case ARRAY_ASSIGNOP_S_X_A_NODE:
        { struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          struct array *arvalue= get_name_arrayptr(node->op3.name_id,newstack + localstack,localbase);
          int inx[100]; /* keep track of indices */
          char *p,*q;
          int j,k;
          char *pspots[MAXARRAYDIMS],*qspots[MAXARRAYDIMS];
          int pstride[MAXARRAYDIMS],qstride[MAXARRAYDIMS];
          int minsize[MAXARRAYDIMS],lastsize;
          REAL scalar;
          int lcount = node[node->left].op5.indexcount;
          int rcount = node[node[node->right].right].op5.indexcount;
          int dim;
          int pint = (alvalue->datatype == INTEGER_TYPE);
          int qint = (arvalue->datatype == INTEGER_TYPE);

          /* Was checked in parser that the arrays have the same number
             of indices, but might be different sizes.  So do
             intersection. */
          /* pop datastarts off stack */
          q = *(char**)(stacktop--);
          scalar = *(stacktop--);
          p = *(char**)(stacktop--);

          dim = arvalue->dim - rcount;
          for ( i = 0 ; i < dim ; i++ )
          { inx[i] = 0;
            minsize[i] = arvalue->sizes[rcount+i] < alvalue->sizes[lcount+i] ?
                          arvalue->sizes[rcount+i] : alvalue->sizes[lcount+i] ;
            pspots[i] = p;
            qspots[i] = q;
          }
          if ( dim >= 2 )
          {
            pstride[dim-2] = alvalue->sizes[alvalue->dim-1]*alvalue->itemsize;
            qstride[dim-2] = arvalue->sizes[arvalue->dim-1]*arvalue->itemsize;
            for ( i = alvalue->dim - 3 ; i >= lcount ; i-- )
              pstride[i] = pstride[i+1]*alvalue->sizes[i+1];
            for ( i = arvalue->dim - 3 ; i >= rcount ; i-- )
              qstride[i] = qstride[i+1]*arvalue->sizes[i+1];
            
          }
          lastsize = minsize[dim-1];

          p = pspots[0];
          q = qspots[0];
          do
          { /* do a row */

                int *pi = (int*)p;
                REAL *pr = (REAL*)p;
                int *qi = (int*)q;
                REAL *qr = (REAL*)q;
                
                 switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) = (int)((qint ? *(qi++) : *(qr++))*scalar);
                       else
                         *(pr++) = (qint ? *(qi++) : *(qr++))*scalar;
                     break;
                  case PLUSASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) += (int)((qint ? *(qi++) : *(qr++))*scalar);
                       else
                         *(pr++) += (qint ? *(qi++) : *(qr++))*scalar;
                     break;
                  case SUBASSIGN_OP:
                    for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) -= (int)((qint ? *(qi++) : *(qr++))*scalar);
                       else
                         *(pr++) -= (qint ? *(qi++) : *(qr++))*scalar;
                     break;                       
                  case MULTASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                         *(pi++) *= (int)((qint ? *(qi++) : *(qr++))*scalar);
                       else
                         *(pr++) *= (qint ? *(qi++) : *(qr++))*scalar;
                     break;
                  case DIVASSIGN_OP:
                    for ( i = 0 ; i < lastsize ; i++ ) 
                       if ( pint )
                       { *pi = (int)(*pi/((qint ? *(qi++) : *(qr++))*scalar));
                          pi++;
                       }
                       else
                         *(pr++) /= (qint ? *(qi++) : *(qr++))*scalar;
                     break;
                } 

            /* increment pointers */
            for ( j = dim-2 ; j >= 0 ; j-- )
            { pspots[j] += pstride[j];
              qspots[j] += qstride[j];
              if ( ++inx[j] < minsize[j] )
              { p = pspots[j];
                q = qspots[j];
                for ( k = j+1 ; k < dim-1 ; k++ )
                { pspots[k] = p;
                  qspots[k] = q;
                }
                break;
              }
              inx[j] = 0 ;
            }  
           } while ( j >= 0 );    
 
          if ( node->flags & RECALC_FLAG )
            recalc_flag = 1;
          if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
            dont_resize_flag = 1;

          break;
        } /* end ARRAY_ASSIGNOP_S_X_A_ */

      case ARRAY_ASSIGNOP_A_P_A_NODE:
      case ARRAY_ASSIGNOP_A_S_A_NODE:
        { 
          struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          struct array *arvalue= get_name_arrayptr(node->op3.name_id,newstack + localstack,localbase);
          struct array *asvalue= get_name_arrayptr(node->op4.name_id,newstack + localstack,localbase);
          int inx[100]; /* keep track of indices */
          char *p,*q,*s;
          int j,k;
          char *pspots[MAXARRAYDIMS],*qspots[MAXARRAYDIMS],*sspots[MAXARRAYDIMS];
          int pstride[MAXARRAYDIMS],qstride[MAXARRAYDIMS],sstride[MAXARRAYDIMS];
          int minsize[MAXARRAYDIMS],lastsize;
          int lcount = node[node->left].op5.indexcount;
          struct treenode *rvalue_node = node + node->right;
          int rcount = rvalue_node[rvalue_node->left].op5.indexcount;
          int scount = rvalue_node[rvalue_node->right].op5.indexcount;
          int dim;
          int pint = (alvalue->datatype == INTEGER_TYPE);
          int qint = (arvalue->datatype == INTEGER_TYPE);
          int sint = (asvalue->datatype == INTEGER_TYPE);
          REAL sign = (node->type==ARRAY_ASSIGNOP_A_P_A_NODE) ? 1.0 : -1.0;

          /* Was checked in parser that the arrays have the same number
             of indices, but might be different sizes.  So do
             intersection. */
          /* pop datastarts off stack */
          s = *(char**)(stacktop--);
          q = *(char**)(stacktop--);
          p = *(char**)(stacktop--);
          
          dim = arvalue->dim - rcount;
          for ( i = 0 ; i < dim ; i++ )
          { inx[i] = 0;
            minsize[i] = arvalue->sizes[rcount+i] < alvalue->sizes[lcount+i] ?
                          arvalue->sizes[rcount+i] : alvalue->sizes[lcount+i] ;
            if ( minsize[i] > asvalue->sizes[scount+i] )
               minsize[i] = asvalue->sizes[scount+i];
            pspots[i] = p;
            qspots[i] = q;
            sspots[i] = s;
          }
          if ( dim >= 2 )
          {
            pstride[dim-2] = alvalue->sizes[dim-1]*alvalue->itemsize;
            qstride[dim-2] = arvalue->sizes[dim-1]*arvalue->itemsize;
            sstride[dim-2] = asvalue->sizes[dim-1]*asvalue->itemsize;
            for ( i = dim - 3 ; i >= 0 ; i-- )
            { pstride[i] = pstride[i+1]*alvalue->sizes[lcount+i+1];
              qstride[i] = qstride[i+1]*arvalue->sizes[rcount+i+1];
              sstride[i] = sstride[i+1]*asvalue->sizes[scount+i+1];
            }
          }
          lastsize = minsize[dim-1];

          p = pspots[0];
          q = qspots[0];
          s = sspots[0];
          do
          { /* do a row */

                int *pi = (int*)p;
                REAL *pr = (REAL*)p;
                int *qi = (int*)q;
                REAL *qr = (REAL*)q;
                int *si = (int*)s;
                REAL *sr = (REAL*)s;
                REAL rvalue;
                
                switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ )
                     { rvalue = (qint ? *(qi++):*(qr++)) + sign*(sint ? *(si++):*(sr++));
                       if ( pint )
                         *(pi++) = (int)rvalue;
                       else
                         *(pr++) = rvalue;
                     }
                     break;
                  case PLUSASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                     { rvalue = (qint ? *(qi++):*(qr++)) + sign*(sint ? *(si++):*(sr++));
                       if ( pint )
                         *(pi++) += (int)rvalue;
                       else
                         *(pr++) += rvalue;
                     }
                     break;
                  case SUBASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                     { rvalue = (qint ? *(qi++):*(qr++)) + sign*(sint ? *(si++):*(sr++));
                       if ( pint )
                         *(pi++) -= (int)rvalue;
                       else
                         *(pr++) -= rvalue;
                     }
                     break;                        
                  case MULTASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                     { rvalue = (qint ? *(qi++):*(qr++)) + sign*(sint ? *(si++):*(sr++));
                       if ( pint )
                         *(pi++) *= (int)rvalue;
                       else
                         *(pr++) *= rvalue;
                     }
                     break;
                  case DIVASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                     { rvalue = (qint ? *(qi++):*(qr++)) + sign*(sint ? *(si++):*(sr++));
                       if ( pint )
                       { *pi = (int)(*pi/rvalue);
                          pi++;
                       }
                       else
                         *(pr++) /= rvalue;
                     }
                     break;
              
                 } 

            /* increment pointers */
            for ( j = dim-2 ; j >= 0 ; j-- )
            { pspots[j] += pstride[j];
              qspots[j] += qstride[j];
              sspots[j] += sstride[j];
              if ( ++inx[j] < minsize[j] )
              { p = pspots[j];
                q = qspots[j];
                s = sspots[j];
                for ( k = j+1 ; k < dim-1 ; k++ )
                { pspots[k] = p;
                  qspots[k] = q;
                  sspots[k] = s;
                 }
                break;
              }
              inx[j] = 0 ;
            }  
           } while ( j >= 0 );    
       
         if ( node->flags & RECALC_FLAG )
           recalc_flag = 1;
         if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
           dont_resize_flag = 1;
       
         } /* end ARRAY_ASSIGNOP_A_P_A_, ARRAY_ASSIGNOP_A_S_A_ */
         break;

      case ARRAY_ADD_NODE:   // result to temp array 
      case ARRAY_SUBTRACT_NODE:
        { 
          struct array *alvalue;
          struct array *arvalue= get_name_arrayptr(node->op3.name_id,newstack + localstack,localbase);
          struct array *asvalue= get_name_arrayptr(node->op4.name_id,newstack + localstack,localbase);
          int inx[100]; /* keep track of indices */
          char *p,*q,*s;
          int j,k;
          char *pspots[MAXARRAYDIMS],*qspots[MAXARRAYDIMS],*sspots[MAXARRAYDIMS];
          int pstride[MAXARRAYDIMS],qstride[MAXARRAYDIMS],sstride[MAXARRAYDIMS];
          int minsize[MAXARRAYDIMS],lastsize;
          int rcount = node[node->left].op5.indexcount;
          int scount = node[node->right].op5.indexcount;
          int dim;
          int qint = (arvalue->datatype == INTEGER_TYPE);
          int sint = (asvalue->datatype == INTEGER_TYPE);
          REAL sign = (node->type==ARRAY_ADD_NODE) ? 1.0 : -1.0;
          struct array **array_info;
          int size;
          int pointercount;
          
          g = globals(node->op2.name_id);
          alvalue = g->attr.arrayptr;  // has dim for temp
          
          /* Was checked in parser that the arrays have the same number
             of indices, but might be different sizes.  But do have
             to check actual sizes now. */
          dim = alvalue->dim;  // common dimension after fixed indices
          for ( i = 0, size = 1, pointercount = 0 ; i < dim ; i++ )
          { if ( arvalue->sizes[rcount+i] != asvalue->sizes[scount+i] )
            { sprintf(errmsg,"Array addition: Arrays have different sizes.\n");
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
              kb_error(5844,errmsg,RECOVERABLE);
            }
            pointercount += size;           
            size *= arvalue->sizes[rcount+i];
          }

          /* Set up temp array for destination */

          if ( !(g->flags & FIXED_SIZE_ARRAY) )
          { alvalue = NULL;
            array_info = (struct array **)(newstack+localstack+g->value.offset);
            if ( array_info )
            { // see if existing storage enough
              if ( *array_info )
              { alvalue = *array_info;
                if ( alvalue->datacount >= size )
                  p = (char*)(alvalue + 1);
                else
                { temp_free((char*)alvalue);
                  alvalue = NULL;
                }
              }
            }
            if ( alvalue == NULL )
            {
              alvalue = *array_info = (struct array*)temp_calloc( sizeof(struct array) +
                 (size+1)*sizeof(REAL)+pointercount*sizeof(REAL*),1);  /* extra for alignment */
              alvalue->flags |= ARRAY_TEMP;
              p = (char*)(alvalue + 1); // for data
            }
            alvalue->dim = dim;
            alvalue->datatype = REAL_TYPE;
            alvalue->itemsize = sizeof(REAL);
            alvalue->datacount = size;
            for ( i = 0 ; i < dim ; i++ )
              alvalue->sizes[i] = arvalue->sizes[rcount+i];
            localbase->flags |= LL_HAS_ARRAY;
          }
          else 
            p = (char*)(newstack+localstack+g->value.offset);

          /* pop datastarts off stack */
          s = *(char**)(stacktop--);
          q = *(char**)(stacktop--);
          
          // push result datastart on stack
          *(REAL**)(++stacktop) = (REAL*)p;  
          
          for ( i = 0 ; i < dim ; i++ )
          { inx[i] = 0;
            minsize[i] = arvalue->sizes[rcount+i];
            pspots[i] = p;
            qspots[i] = q;
            sspots[i] = s;
          }
          if ( dim >= 2 )
          {
            pstride[dim-2] = alvalue->sizes[dim-1]*alvalue->itemsize;
            qstride[dim-2] = arvalue->sizes[dim-1]*arvalue->itemsize;
            sstride[dim-2] = asvalue->sizes[dim-1]*asvalue->itemsize;
            for ( i = dim - 3 ; i >= 0 ; i-- )
            { pstride[i] = pstride[i+1]*alvalue->sizes[i+1];
              qstride[i] = qstride[i+1]*arvalue->sizes[rcount+i+1];
              sstride[i] = sstride[i+1]*asvalue->sizes[scount+i+1];
            }
          }
          lastsize = minsize[dim-1];

          p = pspots[0];
          q = qspots[0];
          s = sspots[0];
          do
          { /* do a row */

            REAL *pr = (REAL*)p;
            int *qi = (int*)q;
            REAL *qr = (REAL*)q;
            int *si = (int*)s;
            REAL *sr = (REAL*)s;

            for ( i = 0 ; i < lastsize ; i++ )
              *(pr++) = (qint ? *(qi++):*(qr++)) + sign*(sint ? *(si++):*(sr++));             

            /* increment pointers */
            for ( j = dim-2 ; j >= 0 ; j-- )
            { pspots[j] += pstride[j];
              qspots[j] += qstride[j];
              sspots[j] += sstride[j];
              if ( ++inx[j] < minsize[j] )
              { p = pspots[j];
                q = qspots[j];
                s = sspots[j];
                for ( k = j+1 ; k < dim-1 ; k++ )
                { pspots[k] = p;
                  qspots[k] = q;
                  sspots[k] = s;
                 }
                break;
              }
              inx[j] = 0 ;
            }  
           } while ( j >= 0 );    
       
         if ( node->flags & RECALC_FLAG )
           recalc_flag = 1;
          if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
            dont_resize_flag = 1;

         } /* end ARRAY_ADD_NODE */
         break;

      case ARRAY_SCALAR_MULTIPLY_NODE:
      case ARRAY_SCALAR_DIVIDE_NODE:
        { // scalar multiply to temp array.  Scalar pushed on stack first.
          struct array *alvalue;
          struct array *arvalue= get_name_arrayptr(node->op3.name_id,newstack + localstack,localbase);
          REAL scalar;
          int inx[100]; /* keep track of indices */
          char *p,*q;
          int j,k;
          char *pspots[MAXARRAYDIMS],*qspots[MAXARRAYDIMS];
          int pstride[MAXARRAYDIMS],qstride[MAXARRAYDIMS];
          int minsize[MAXARRAYDIMS],lastsize;
          int rcount = node[node->right].op5.indexcount;
          int dim;
          int qint = (arvalue->datatype == INTEGER_TYPE);
          struct array **array_info;
          int size;
          int pointercount;
          
          g = globals(node->op2.name_id);
          alvalue = g->attr.arrayptr;  // has dim for temp
          
          dim = alvalue->dim;  // common dimension after fixed indices
          for ( i = 0, size = 1, pointercount = 0 ; i < dim ; i++ )
          { 
            pointercount += size;           
            size *= arvalue->sizes[rcount+i];
          }

          /* Set up temp array for destination */

          if ( !(g->flags & FIXED_SIZE_ARRAY) )
          { alvalue = NULL;
            array_info = (struct array **)(newstack+localstack+g->value.offset);
            if ( array_info )
            { // see if existing storage enough
              if ( *array_info )
              { alvalue = *array_info;
                if ( alvalue->datacount >= size )
                  p = (char*)(alvalue + 1);
                else
                { temp_free((char*)alvalue);
                  alvalue = NULL;
                }
              }
            }
            if ( alvalue == NULL )
            { alvalue = *array_info = (struct array*)temp_calloc( sizeof(struct array) +
                 (size+1)*sizeof(REAL)+pointercount*sizeof(REAL*),1);  /* extra for alignment */  
              alvalue->flags |= ARRAY_TEMP;
              p = (char*)(alvalue + 1); // for data
            }
            alvalue->dim = dim;
            alvalue->datatype = REAL_TYPE;
            alvalue->itemsize = sizeof(REAL);
            alvalue->datacount = size;
            for ( i = 0 ; i < dim ; i++ )
              alvalue->sizes[i] = arvalue->sizes[rcount+i];
            localbase->flags |= LL_HAS_ARRAY;
          }
          else 
            p = (char*)(newstack+localstack+g->value.offset);

          /* pop datastarts off stack */
          q = *(char**)(stacktop--);
          scalar = *(stacktop--);
          if ( node->type == ARRAY_SCALAR_DIVIDE_NODE )
          { if ( scalar == 0.0 ) 
            { sprintf(errmsg,"Dividing array by zero in %s.\n",ex_current.name);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
              kb_error(5258,errmsg,RECOVERABLE);
            }
            else
              scalar = 1/scalar;
          }
          
          // push result datastart on stack
          *(REAL**)(++stacktop) = (REAL*)p;  
          
          for ( i = 0 ; i < dim ; i++ )
          { inx[i] = 0;
            minsize[i] = arvalue->sizes[rcount+i];
            pspots[i] = p;
            qspots[i] = q;
          }
          if ( dim >= 2 )
          {
            pstride[dim-2] = alvalue->sizes[dim-1]*alvalue->itemsize;
            qstride[dim-2] = arvalue->sizes[dim-1]*arvalue->itemsize;
            for ( i = dim - 3 ; i >= 0 ; i-- )
            { pstride[i] = pstride[i+1]*alvalue->sizes[i+1];
              qstride[i] = qstride[i+1]*arvalue->sizes[rcount+i+1];
            }
          }
          lastsize = minsize[dim-1];

          p = pspots[0];
          q = qspots[0];
          do
          { /* do a row */

            REAL *pr = (REAL*)p;
            int *qi = (int*)q;
            REAL *qr = (REAL*)q;

            for ( i = 0 ; i < lastsize ; i++ )
                *(pr++) = scalar*(qint ? *(qi++):*(qr++)); 
 
            /* increment pointers */
            for ( j = dim-2 ; j >= 0 ; j-- )
            { pspots[j] += pstride[j];
              qspots[j] += qstride[j];
              if ( ++inx[j] < minsize[j] )
              { p = pspots[j];
                q = qspots[j];
                for ( k = j+1 ; k < dim-1 ; k++ )
                { pspots[k] = p;
                  qspots[k] = q;
                 }
                break;
              }
              inx[j] = 0 ;
            }  
           } while ( j >= 0 );    
       
         if ( node->flags & RECALC_FLAG )
           recalc_flag = 1;
         if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
            dont_resize_flag = 1;

         } /* end ARRAY_SCALAR_MULTIPLY_NODE */
         break;

      
      case ARRAY_MULTIPLY_NODE:
        { /* standard matrix multiplication, result in temp matrix */
          /* Like P[I][J][K] = Q[I][J][M]*R[M][K] */
          struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          struct array *arvalue= get_name_arrayptr(node->op3.name_id,newstack + localstack,localbase);
          struct array *asvalue= get_name_arrayptr(node->op4.name_id,newstack + localstack,localbase);
          int inx[MAXARRAYDIMS]; /* keep track of indices */
          char *p,*q,*s;
          int j,k;
          char *pspots[MAXARRAYDIMS],*qspots[MAXARRAYDIMS],*sspots[MAXARRAYDIMS];
          int pstride[MAXARRAYDIMS],qstride[MAXARRAYDIMS],sstride[MAXARRAYDIMS];
          int lastsize;
          struct array **array_info; // temp array struct pointer on stack
          int rcount = node[node->left].op5.indexcount;
          int scount = node[node->right].op5.indexcount;
          int dim,ldim,sdim,rdim;
 
          /* Set up temp array for destination */
          g = globals(node->op2.name_id);
          alvalue = g->attr.arrayptr;
          dim = alvalue->dim;
          if ( !(g->flags & FIXED_SIZE_ARRAY) )
          { struct array ta;    
            int size,pointercount; 
            for ( i=0,size=1,pointercount=0 ;
              i + node[node->left].op5.indexcount < arvalue->dim-1 ; i++ )
            { pointercount += size;
              ta.sizes[i] = arvalue->sizes[node[node->left].op5.indexcount + i];
              size *= ta.sizes[i];
            }
            for ( j = 1 ;
              j + node[node->right].op5.indexcount < asvalue->dim ; j++,i++ )
            { pointercount += size;
              ta.sizes[i] = asvalue->sizes[node[node->right].op5.indexcount + j];
              size *= ta.sizes[i];
            }

            alvalue = NULL;
            array_info = (struct array **)(newstack+localstack+g->value.offset);
            if ( array_info )
            { // see if existing storage enough
              if ( *array_info )
              { alvalue = *array_info;
                if ( alvalue->datacount >= size )
                  p = (char*)(alvalue + 1);
                else
                { temp_free((char*)alvalue);
                  alvalue = NULL;
                }
              }
            }
            if ( alvalue == NULL )
            { alvalue = *array_info = (struct array*)temp_calloc( sizeof(struct array) +
                 (size+1)*sizeof(REAL)+pointercount*sizeof(REAL*),1);  /* extra for alignment */
              alvalue->flags |= ARRAY_TEMP;
              p = (char*)(alvalue + 1); // for data
            }
            alvalue->dim = dim;
            alvalue->datatype = REAL_TYPE;
            alvalue->itemsize = sizeof(REAL);
            alvalue->datacount = size;
            for ( i = 0 ; i < dim ; i++ )
              alvalue->sizes[i] = ta.sizes[rcount+i];
            localbase->flags |= LL_HAS_ARRAY;
          }
          else 
            p = (char*)(newstack+localstack+g->value.offset);

          /* pop datastarts off stack */
          s = *(char**)(stacktop--);
          q = *(char**)(stacktop--);
          
           // push result datastart on stack
          *(REAL**)(++stacktop) = (REAL*)p; 
          
          ldim = alvalue->dim;
          rdim = arvalue->dim - rcount;
          sdim = asvalue->dim - scount;

          for ( i = 0 ; i < ldim ; i++ ) 
          { pspots[i] = p;
            inx[i] = 0;
          }
          for ( i = 0 ; i < rdim ; i++ ) 
            qspots[i] = q;
          for ( i = 0 ; i < sdim ; i++ ) 
            sspots[i] = s;
          
          if ( ldim >= 1 )
          { pstride[ldim-1] = alvalue->itemsize;
            for ( i = ldim - 2 ; i >= 0 ; i-- )
              pstride[i] = pstride[i+1]*alvalue->sizes[i+1];
          }
          if ( rdim >= 1 )
          {
            qstride[rdim-1] = arvalue->itemsize;            
            for ( i = rdim - 2 ; i >= 0 ; i-- )
              qstride[i] = qstride[i+1]*arvalue->sizes[rcount+i+1];              
          }
          if ( sdim >= 1 )
          {
            sstride[sdim-1] = asvalue->itemsize;
            for ( i = sdim - 2 ; i >= 0 ; i-- )
              sstride[i] = sstride[i+1]*asvalue->sizes[scount+i+1];            
          }
          lastsize = arvalue->sizes[arvalue->dim-1];
          if ( lastsize != asvalue->sizes[scount] )
          { sprintf(errmsg,"Mismatched sizes in matrix multiplication, %d vs %d\n",
                lastsize,asvalue->sizes[scount] );
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
            kb_error(1916,errmsg,RECOVERABLE);
          }

          p = pspots[0];
          q = qspots[0];
          s = sspots[0];
          do
          { 
            REAL *pp = (REAL*)p;
            char *qq = q;
            char *ss = s;
            *pp = 0.0;
             if ( (arvalue->datatype == INTEGER_TYPE) && (asvalue->datatype == INTEGER_TYPE) )
               for ( i = 0 ; i < lastsize ; i++ ) 
               { *pp += (*(int*)qq)*(*(int*)ss);
                  qq += sizeof(int);
                  ss += sstride[0];
               }
             else if ( arvalue->datatype == INTEGER_TYPE )
               for ( i = 0 ; i < lastsize ; i++ ) 
               { *pp += (*(int*)qq)*(*(REAL*)ss);
                  qq += sizeof(int);
                  ss += sstride[0];
               }
            else if ( asvalue->datatype == INTEGER_TYPE )
               for ( i = 0 ; i < lastsize ; i++ ) 
               { *pp += (*(REAL*)qq)*(*(int*)ss);
                  qq += sizeof(REAL);
                  ss += sstride[0];
               }
            else 
               for ( i = 0 ; i < lastsize ; i++ ) 
               { *pp += (*(REAL*)qq)*(*(REAL*)ss);
                  qq += sizeof(REAL);
                  ss += sstride[0];
               }

            /* increment pointers */
            for ( j = ldim-1 ; j >= 0 ; j-- )
            { pspots[j] += pstride[j];
              if ( j < rdim-1 )
                qspots[j] += qstride[j];
              else
                sspots[j-rdim+2] += sstride[j-rdim+2];
              if ( ++inx[j] < alvalue->sizes[j] )
              { p = pspots[j];                
                q = qspots[j];
                s = sspots[j-rdim+2];
                for ( k = j+1 ; k < ldim ; k++ )
                  pspots[k] = p;
                for ( k = j+1 ; k < rdim ; k++ )
                  qspots[k] = q;
                for ( k = j-rdim+2 ; k < sdim ; k++ )
                  sspots[k] = s;
                 
                break;
              }
              inx[j] = 0 ;
            }  
           } while ( j >= 0 );         

          if ( node->flags & RECALC_FLAG )
            recalc_flag = 1;
          if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
            dont_resize_flag = 1;
     

         } /* end ARRAY_MULTIPLY_NODE */
         break;

      case ARRAY_ASSIGNOP_A_X_A_NODE:
        { /* standard matrix multiplication */
          /* Like P[I][J][K] = Q[I][J][M]*R[M][K] */
          struct array *alvalue= get_name_arrayptr(node->op2.name_id,newstack + localstack,localbase);
          struct array *arvalue= get_name_arrayptr(node->op3.name_id,newstack + localstack,localbase);
          struct array *asvalue= get_name_arrayptr(node->op4.name_id,newstack + localstack,localbase);
          int inx[MAXARRAYDIMS]; /* keep track of indices */
          char *p,*q,*s;
          int j,k;
          char *pspots[MAXARRAYDIMS],*qspots[MAXARRAYDIMS],*sspots[MAXARRAYDIMS];
          int pstride[MAXARRAYDIMS],qstride[MAXARRAYDIMS],sstride[MAXARRAYDIMS];
          int lastsize;
          int lcount = node[node->left].op5.indexcount;
          struct treenode *rvalue_node = node + node->right;
          int rcount = rvalue_node[rvalue_node->left].op5.indexcount;
          int scount = rvalue_node[rvalue_node->right].op5.indexcount;

          int ldim,sdim,rdim;
          REAL *ptemp,*rtemp,*stemp;
          int *oldp; /* destination integer array */
 
          /* Was checked in parser that the arrays have the same number
             of indices, but might be different sizes.  So do
             intersection. */
          /* pop datastarts off stack */
          s = *(char**)(stacktop--);
          q = *(char**)(stacktop--);
          p = *(char**)(stacktop--);
          
          ldim = alvalue->dim - lcount;
          rdim = arvalue->dim - rcount;
          sdim = asvalue->dim - scount;

          /* Temp arrays in case of integer data */
          if ( alvalue->datatype == INTEGER_TYPE )
          { int count;
            for ( count = 1, j = 0 ; j < ldim ; j++ )
                count *= alvalue->sizes[lcount+j];
            ptemp = (REAL*)temp_calloc(count,sizeof(REAL));
            oldp = (int*)p;
            if ( node->op1.assigntype != ASSIGN_OP )
              for ( j = 0 ; j < count ; j++ )
                ptemp[j] = ((int*)p)[j];
            p = (char*)ptemp;
          }

          /* Temp arrays in case of integer data */
          if ( arvalue->datatype == INTEGER_TYPE )
          { int count;
            for ( count = 1, j = 0 ; j < rdim ; j++ )
                count *= arvalue->sizes[rcount+j];
            rtemp = (REAL*)temp_calloc(count,sizeof(REAL));
            for ( j = 0 ; j < count ; j++ )
              rtemp[j] = ((int*)q)[j];
            q = (char*)rtemp;
          }

          /* Temp arrays in case of integer data */
          if ( asvalue->datatype == INTEGER_TYPE )
          { int count;
            for ( count = 1, j = 0 ; j < sdim ; j++ )
                count *= asvalue->sizes[scount+j];
            stemp = (REAL*)temp_calloc(count,sizeof(REAL));
            for ( j = 0 ; j < count ; j++ )
              stemp[j] = ((int*)s)[j];
            s = (char*)stemp;
          }

          for ( i = 0 ; i < ldim ; i++ ) 
          { pspots[i] = p;
            inx[i] = 0;
          }
          for ( i = 0 ; i < rdim ; i++ ) 
            qspots[i] = q;
          for ( i = 0 ; i < sdim ; i++ ) 
            sspots[i] = s;
          
          if ( ldim >= 1 )
          { pstride[ldim-1] = sizeof(DOUBLE);
            for ( i = ldim - 2 ; i >= 0 ; i-- )
              pstride[i] = pstride[i+1]*alvalue->sizes[lcount+i+1];
          }
          if ( rdim >= 1 )
          {
            qstride[rdim-1] = sizeof(DOUBLE);            
            for ( i = rdim - 2 ; i >= 0 ; i-- )
              qstride[i] = qstride[i+1]*arvalue->sizes[rcount+i+1];              
          }
         if ( sdim >= 1 )
          {
            sstride[sdim-1] = sizeof(DOUBLE);
            for ( i = sdim - 2 ; i >= 0 ; i-- )
              sstride[i] = sstride[i+1]*asvalue->sizes[scount+i+1];            
          }
          lastsize = arvalue->sizes[arvalue->dim-1];
          if ( lastsize != asvalue->sizes[scount] )
          { sprintf(errmsg,"Mismatched sizes in matrix multiplication, %d vs %d.\n",lastsize,
               asvalue->sizes[scount]);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
            kb_error(1917,errmsg,RECOVERABLE);
          }
          for ( j = 0 ; j < rdim-1 ; j++ )
          { if ( alvalue->sizes[lcount+j] != arvalue->sizes[rcount+j] )
            { sprintf(errmsg,"Mismatched sizes in matrix multiplication.  \n");
              sprintf(errmsg+strlen(errmsg),"Index %d of left side is %d, index %d of first factor is %d\n",
                lcount+j+1,alvalue->sizes[lcount+j],rcount+j+1,arvalue->sizes[rcount+j]);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
              kb_error(4530,errmsg,RECOVERABLE);
            }
          }
          for ( j = 0 ; j < scount-1 ; j++ )
          { if ( alvalue->sizes[lcount+rdim+j] != asvalue->sizes[scount+j+1] )
            { sprintf(errmsg,"Mismatched sizes in matrix multiplication.\n");
              sprintf(errmsg+strlen(errmsg),"Index %d of left side is %d, index %d of second factor is %d\n",
                lcount+rdim+j+1,alvalue->sizes[lcount+j],scount+j+1,asvalue->sizes[scount+j]);
              sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
              kb_error(1918,errmsg,RECOVERABLE);
            }
          }

          
          p = pspots[0];
          q = qspots[0];
          s = sspots[0];
          do
          { 
            
                REAL *pp = (REAL*)p;
                REAL *qq = (REAL*)q;
                REAL *ss = (REAL*)s;
                switch ( node->op1.assigntype )
                { case ASSIGN_OP:
                     { *pp = 0.0;
                        for ( i = 0 ; i < lastsize ; i++ ) 
                        { *pp += (*qq)*(*ss);
                           qq++;
                           ss = (REAL*)((char*)ss+sstride[0]);
                        }
                      }
                     break;
                  case PLUSASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) += *(qq++) + *(ss++);
                     break;
                  case SUBASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) -= *(qq++) + *(ss++);
                     break;                        
                  case MULTASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) *= *(qq++) + *(ss++);
                     break;
                  case DIVASSIGN_OP:
                     for ( i = 0 ; i < lastsize ; i++ ) 
                       *(pp++) /= *(qq++) + *(ss++);
                     break;
                }
      

            /* increment pointers */
            for ( j = ldim-1 ; j >= 0 ; j-- )
            { pspots[j] += pstride[j];
              if ( j < rdim-1 )
                qspots[j] += qstride[j];
              else
                sspots[j-rdim+2] += sstride[j-rdim+2];
              if ( ++inx[j] < alvalue->sizes[lcount+j] )
              { p = pspots[j];                
                q = qspots[j];
                s = sspots[j-rdim+2];
                for ( k = j+1 ; k < ldim ; k++ )
                  pspots[k] = p;
                for ( k = j+1 ; k < rdim ; k++ )
                  qspots[k] = q;
                for ( k = j-rdim+2 ; k < sdim ; k++ )
                  sspots[k] = s;
                 
                break;
              }
              inx[j] = 0 ;
            }  
           } while ( j >= 0 );    
       
          /* Temp arrays in case of integer data */
          if ( alvalue->datatype == INTEGER_TYPE )
          { int count;
            for ( count = 1, j = 0 ; j < ldim ; j++ )
                count *= alvalue->sizes[lcount+j];
            for ( j = 0 ; j < count ; j++ )
              oldp[j] = (int)(ptemp[j]);
            temp_free((char*)ptemp);
          }
          if ( arvalue->datatype == INTEGER_TYPE )
            temp_free((char*)rtemp);
          if ( asvalue->datatype == INTEGER_TYPE )
            temp_free((char*)stemp);


         if ( node->flags & RECALC_FLAG )
           recalc_flag = 1;
         if ( node->flags & DONT_RESIZE_FLAG )  // for modifying view_matrix before 's'
           dont_resize_flag = 1;

       
         } /* end ARRAY_ASSIGNOP_A_X_A_ */
         break;

     /* end whole-array syntax */
    
      case PLUSASSIGN_NODE:
      case SUBASSIGN_NODE:
      case MULTASSIGN_NODE:
      case DIVASSIGN_NODE:
      case SET_GLOBAL_NODE:
       { struct global *g = globals(node->op1.name_id);
         REAL *x;

         if ( g->flags & GLOB_LOCALVAR )
            x = newstack + localstack + g->value.offset;
         else if ( g->flags & ORDINARY_PARAM )
            x = &g->value.real;
         else 
         { sprintf(errmsg, "Variable %s is read-only.\n",g->name);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(3455,errmsg,WARNING);
           break;
         }

         if ( g->flags & STRINGVAL )
         { if ( g->flags & GLOB_LOCALVAR )
             myfree(*(char**)x); 
           else  myfree(g->value.string);
           g->flags &= ~STRINGVAL; 
         }

         switch(node->type)
         { 
            case SET_GLOBAL_NODE: *x = *(stacktop--); break;
            case PLUSASSIGN_NODE: *x += *(stacktop--); break;
            case SUBASSIGN_NODE: *x -= *(stacktop--); break;
            case MULTASSIGN_NODE: *x *= *(stacktop--); break;
            case DIVASSIGN_NODE: 
              if ( *stacktop == 0.0 )
              { sprintf(errmsg,"Division by zero.\n");
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                   file_names[node->file_no],node->line_no);
                kb_error(2562,errmsg,RECOVERABLE);
              }
              *x /= *(stacktop--); break;
         }
         if ( g->attr.varstuff.on_assign_call )
         { struct  global *gg = globals(g->attr.varstuff.on_assign_call);
           PUSH_TRACE;
           eval(&gg->value.proc,NULL,NULLID,NULL);
           POP_TRACE;
         }
         if ( g->flags & RECALC_PARAMETER )
            recalc_flag = 1;

       }
       break;

      case PRE_INCREMENT_NODE: case POST_INCREMENT_NODE:
       { struct global *g = globals(node->op1.name_id);
         REAL *x;

         if ( g->flags & GLOB_LOCALVAR )
            x = newstack + localstack + g->value.offset;
         else if ( g->flags & ORDINARY_PARAM )
            x = &g->value.real;
         else 
         { sprintf(errmsg, "Variable %s is read-only.\n",g->name);
           sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
           kb_error(4455,errmsg,WARNING);
           break;
         }

         switch(node->op2.assigntype)
         { 
            case PLUSPLUS_OP: *x += 1; break;
            case MINUSMINUS_OP: *x -= 1; break;
         }
         if ( g->flags & RECALC_PARAMETER )
            recalc_flag = 1;
       }
       break;

      case SET_PERM_GLOBAL_NODE:
       { struct global *g = perm_globals(node->op1.name_id);
         REAL *x;

         if ( g->flags & STRINGVAL )
         { free(g->value.string); g->flags &= ~STRINGVAL; }

         x = &g->value.real;
         *x = *(stacktop--); 
         if ( g->flags & RECALC_PARAMETER )
            recalc_flag = 1;
       }
       break;




      case FINISHED_NODE:
         if ( localbase && (localbase->flags & LL_HAS_ARRAY) )
          { /* de-allocate local array storage */
            int n;
            for ( n = 0 ; n < localbase->count ; n++ )
            { struct global *g = &(localbase->list[n].g);
              if ( (g->flags & ARRAY_PARAM) && !(g->flags & FIXED_SIZE_ARRAY) )
              { struct array *a = *(struct array**)(newstack + localstack + g->value.offset);
                if ( a ) temp_free((char*)a);
                a = NULL;
              }
            }
         }
        if ( this_frame->flags & BASE_OF_EVAL )
        { /* want to exit from eval() */
          goto the_exit; 
        }       
        else /* want to return within eval() */
        {          
          return_value = *stacktop;
          stacktop = (REAL*)this_frame;
          stacktop--; /* since points to occupied spot */
          node = this_frame->return_node;
          td->frame_spot = this_frame->parent_frame_spot;
          localstack = (int)((REAL*)(this_frame)+FRAME_SPACE - newstack);
          ex_current = this_frame->base_ex;
          localbase = ex_current.locals;
          if ( ex_current.locals )
            localcount = ex_current.locals->totalsize;
          else localcount = 0;
        }      
        break;

    default:
        other_stuff(node,&recalc_flag,
            &update_display_flag,q_id,newstack + localstack,localbase);
        break;
      
    }      

    if ( breakflag && !hessian_iterate_flag && (!threadflag || (this_thread == main_thread_id)) )
    {
      switch ( breakflag )
      { case BREAKFULL:
        { sprintf(errmsg,"Command aborted due to user interrupt.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(2073,errmsg, RECOVERABLE);
        }
        break;
         
      case BREAKABORT:
        { sprintf(errmsg,"Command aborted.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
          kb_error(3234,errmsg, RECOVERABLE_ABORT);
        }
        break;
      case BREAKAFTERWARNING: 
        { sprintf(errmsg,"Command aborted due to break_after_warning.\n");
          sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                file_names[node->file_no],node->line_no);
          kb_error(3109,errmsg, RECOVERABLE);
        }
      }
    }

#ifdef _XXDEBUG
   { static int didwarn;
    /* Check predicted stack alignment.  There are circumstances when alignment
       is not equal (e.g. skipping procedure bodies when assigning them,
       so just check there is room. */
    if ( (node[1].type != SETUP_FRAME_NODE) &&
            (stacktop > (REAL*)this_frame + FRAME_SPACE + localcount + node->stack_spot) )
    { sprintf(errmsg,"Intermediate stack misalignment after node type %d.\n",
         node->type);
      sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
      if ( ! didwarn )  /* to prevent error cascade */
         kb_error(2671,errmsg,WARNING);
      didwarn = 1;
    }
    else didwarn = 0;
   }
#endif

if ( td->stack_top < td->eval_stack )
  kb_error(5243,"BAD EVAL STACK\n",RECOVERABLE);
  
    if ( newstack[stackmax-1] != STACKMAGIC ) /* nearing overflow */
    { kb_error(3488,"Evaluation stack overflow.\n",RECOVERABLE);
    }
  }
the_exit:

 
  if ( !params )
  { if ( web.counts_changed && (this_thread == main_thread_id) )
       flush_counts();

    if ( recalc_flag && autorecalc_flag ) recalc();
    else if ( update_display_flag ) update_display();
  }
  PROF_EVAL_END(ex_original);
  iterate_flag = old_flag; 
  if ( stacktop == (REAL*)this_frame + FRAME_SPACE + localcount  ) 
  { /* Have return value on stack */
    REAL retval = *(stacktop--);
    stacktop = (REAL*)this_frame;
    stacktop--; /* since points to occupied spot */
    td->frame_spot = this_frame->parent_frame_spot;
    return retval; 
  }
  else if ( stacktop != (REAL*)this_frame + FRAME_SPACE + localcount - 1)
  { 
    sprintf(errmsg,"Internal error: Stack misalignment by %d in eval()\n",
       (int)(stacktop-(REAL*)this_frame-localcount-FRAME_SPACE+1));
    sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
             file_names[node->file_no],node->line_no);
    kb_error(1298,errmsg, RECOVERABLE);
  }
  
  stacktop = (REAL*)this_frame;
  stacktop--; /* since points to occupied spot */
  td->frame_spot = this_frame->parent_frame_spot;
  return 0.0;
} /* end eval() */

/************************************************************************
*
* function: flush_counts()
*
* purpose: print pending counts
*/

void flush_counts()
{
     if ( web.counts_changed & fix_count_bit )
     { sprintf(msg,"Fixed: %d\n",web.fix_count); outstring(msg); }

     if ( web.counts_changed & unfix_count_bit )
     { sprintf(msg,"Unfixed: %d\n",web.unfix_count); outstring(msg); }

     if ( web.counts_changed & equi_count_bit )
     { sprintf(msg,"Edges equiangulated: %d\n",web.equi_count); outstring(msg); }

     if ( web.counts_changed & edge_delete_count_bit )
     { sprintf(msg,"Edges deleted: %d\n",web.edge_delete_count); outstring(msg); }

     if ( web.counts_changed & facet_delete_count_bit )
     { sprintf(msg,"Facets deleted: %d\n",web.facet_delete_count); outstring(msg); }

     if ( web.counts_changed & edge_refine_count_bit )
     { sprintf(msg,"Edges refined: %d\n",web.edge_refine_count); outstring(msg); }

     if ( web.counts_changed & facet_refine_count_bit )
     { sprintf(msg,"Facets refined: %d\n",web.facet_refine_count); outstring(msg); }

     if ( web.counts_changed & vertex_dissolve_count_bit )
     { sprintf(msg,"Vertices dissolved: %d\n",web.vertex_dissolve_count); outstring(msg); }

     if ( web.counts_changed & edge_dissolve_count_bit )
     { sprintf(msg, "Edges dissolved: %d\n",web.edge_dissolve_count); outstring(msg); }

     if ( web.counts_changed & facet_dissolve_count_bit )
     { sprintf(msg,"Facets dissolved: %d\n",web.facet_dissolve_count); outstring(msg); }

     if ( web.counts_changed & body_dissolve_count_bit )
     { sprintf(msg,"Bodies dissolved: %d\n",web.body_dissolve_count); outstring(msg); }

     if ( web.counts_changed & edge_reverse_count_bit )
     { sprintf(msg, "Edges reversed: %d\n",web.edge_reverse_count); outstring(msg); }

     if ( web.counts_changed & facet_reverse_count_bit )
     { sprintf(msg, "Facets reversed: %d\n",web.facet_reverse_count); outstring(msg); }

     if ( web.counts_changed & vertex_pop_count_bit )
     { sprintf(msg,"Vertices popped: %d\n",web.vertex_pop_count); outstring(msg); }

     if ( web.counts_changed & edge_pop_count_bit )
     { sprintf(msg,"Edges popped: %d\n",web.edge_pop_count); outstring(msg); }

     if ( web.counts_changed & pop_tri_to_edge_count_bit )
     { sprintf(msg,"pop_tri_to_edge count: %d\n",web.pop_tri_to_edge_count); outstring(msg); }

     if ( web.counts_changed & pop_edge_to_tri_count_bit )
     { sprintf(msg,"pop_edge_to_tri count: %d\n",web.pop_edge_to_tri_count); outstring(msg); }

     if ( web.counts_changed & pop_quad_to_quad_count_bit )
     { sprintf(msg,"pop_quad_to_quad count: %d\n",web.pop_quad_to_quad_count); outstring(msg); }

     if ( web.counts_changed & edgeswap_count_bit )
     { sprintf(msg,"Edges swapped: %d\n",web.edgeswap_count); outstring(msg); }

     if ( web.counts_changed & t1_edgeswap_count_bit )
     { sprintf(msg,"T1 swaps: %d\n",web.t1_edgeswap_count); outstring(msg); }


     web.counts_reported = ~0;
     web.counts_changed  = 0;

}  // end flush_counts()


/************************************************************************
*
* function: reset_counts()
*
* purpose: set counts to 0 for event counts and profiling counts.
*/

void reset_counts()
{
     web.equi_count = 0;
     web.edge_delete_count = 0;
     web.facet_delete_count = 0;
     web.edge_refine_count = 0;
     web.facet_refine_count = 0;
     web.notch_count = 0;
     web.vertex_dissolve_count = 0;
     web.edge_dissolve_count = 0;
     web.facet_dissolve_count = 0;
     web.body_dissolve_count = 0;
     web.edge_reverse_count = 0;
     web.facet_reverse_count = 0;
     web.vertex_pop_count = 0;
     web.edge_pop_count = 0;
     web.pop_tri_to_edge_count = 0;
     web.pop_edge_to_tri_count = 0;
     web.pop_quad_to_quad_count = 0;
     web.where_count = 0;
     web.edgeswap_count = 0;
     web.fix_count = 0;
     web.unfix_count = 0;
     web.t1_edgeswap_count = 0;
     web.notch_count = 0;

     web.counts_reported = 0;
     web.counts_changed  = 0; 

#ifdef PROFILING_ENABLED
   { int i;
     for ( i = LOW_INST ; i < meth_inst_count ; i++ )
     { struct method_instance *mi = METH_INSTANCE(i);
       mi->value_call_count = mi->grad_call_count = mi->hess_call_count = 0;
       mi->value_elapsed_time = mi->grad_elapsed_time 
         = mi->hess_elapsed_time = 0.0;
     }
     for ( i = 0 ; i < 2 ; i++ )
     { element_setup_elapsed_time[i] = 0;
       calc_quants_elapsed_time[i] = 0;
       calc_quant_grads_elapsed_time[i] = 0;
       calc_quant_hess_elapsed_time[i] = 0;
       exparse_elapsed_time[i] = 0;
       yyparse_elapsed_time[i] = 0;
       yylex_elapsed_time[i] = 0;
       kblex_elapsed_time[i] = 0;
       hessian_solve_elapsed_time[i] = 0;
       hessian_mul_elapsed_time[i] = 0;
       hessian_AIJ_setup_elapsed_time[i] = 0;
       hessian_constraint_setup_elapsed_time[i] = 0;
       hessian_project_setup_elapsed_time[i] = 0;
       hessian_factor_elapsed_time[i] = 0;
       hessian_CHinvC_elapsed_time[i] = 0;
     }
     find_cpu_speed();
   }
#endif
} // end reset_counts()

/****************************************************************************
*
* Function: get_indexed_subtype()
*
* Purpose: Implements INDEXED_SUBTYPE_NODE for eval_tree(), eval_all(),
*          and eval_sec().
* 
* Return: Id of desired element.
*/

element_id get_indexed_subtype(struct treenode *node, 
                               int ord, 
                               element_id parent,
                               struct expnode *ex)
{ element_id next_id = NULLID;
  element_id id;
  int n,i;
  facetedge_id fe;

          int ptype = id_type(parent);
          element_id first; /* sentinel for looping */

          if ( ord < 0 )
          { sprintf(errmsg,"Element index must be positive in %s.\n",ex->name);
            sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
            kb_error(1275,errmsg,RECOVERABLE);
          }
          id=NULLID;
          switch ( ptype )
          { 
             case VERTEX:
             switch ( node->op1.eltype )/* subtype */ 
             { 
               case EDGE:  /* indexed edge of vertex */
               id = first = get_vertex_edge(parent);
               if ( !valid_element(id) )
               { sprintf(errmsg,"Vertex %s has no edges.\n",ELNAME(parent));
                 sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                 kb_error(1276,errmsg,RECOVERABLE);
               }
               for ( n = 1 ; n <= ord ; n++ )
               { id = get_next_tail_edge(id);
                 if ( equal_id(id,first) )
                 { sprintf(errmsg,"Edge index %d exceeds valence of vertex %s.\n",
                     ord+1,ELNAME(id));
                   sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                      file_names[node->file_no],node->line_no);
                   kb_error(1277,errmsg,RECOVERABLE);
                 }
               }
               next_id = id;
               break;
              
              case FACET: /* indexed facet of vertex */

              id = first = get_vertex_first_facet(parent);
              if ( !valid_element(id) )
              { sprintf(errmsg,"Vertex %s has no facets.\n",ELNAME(parent));
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
                kb_error(1278,errmsg,RECOVERABLE);
              }
              for ( n = 1 ; n <= ord ; n++ )
              { id = get_next_vertex_facet(parent,id);
                if ( equal_id(id,first) )
                { sprintf(errmsg,
                    "Facet index %d exceeds facet valence of vertex %s.\n",
                       ord+1,ELNAME(parent));
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                    file_names[node->file_no],node->line_no);
                  kb_error(1279,errmsg,RECOVERABLE);
                }
              }
              if ( id_type(id) == FACETEDGE )
                id = get_fe_facet(id);
              next_id = positive_id(id);
              break;

             }
             break;

         case EDGE:
            switch ( node->op1.eltype /* subtype */ )
            { case VERTEX:
              if ( ord >= web.skel[EDGE].ctrlpts )
              { sprintf(errmsg,
                 "Index, %d, exceeds the number of vertices on an edge, %d.\n",
                     ord+1,web.skel[EDGE].ctrlpts);
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                   file_names[node->file_no],node->line_no);
                kb_error(1280,errmsg, RECOVERABLE);
              } 
              if ( web.modeltype == LAGRANGE )
              { next_id = get_edge_vertices(parent)
                  [inverted(parent) ? web.skel[EDGE].ctrlpts-ord-1 : ord]; 
              }
              else 
                 switch ( ord )
                 { case 0: next_id = get_edge_tailv(parent); break;
                   case 1: next_id = get_edge_headv(parent); break;
                   case 2: next_id = get_edge_midv(parent); break;
                 } 
              break;

            case FACET:
              id = first = get_edge_fe(parent);
              if ( !valid_element(id) )
              { sprintf(errmsg,"Edge %s has no facets.\n",ELNAME(parent));
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                 file_names[node->file_no],node->line_no);
                kb_error(1176,errmsg,RECOVERABLE);
              }
              for ( n = 1 ; n <= ord ; n++ )
              { id = get_next_facet(id);
                if ( equal_id(id,first) )
                { sprintf(errmsg,
                     "Facet index, %d, exceeds number of facets on edge %s.\n",
                          ord+1,ELNAME(parent));
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                       file_names[node->file_no],node->line_no);
                  kb_error(1186,errmsg,RECOVERABLE);
                }

              }
             next_id = get_fe_facet(id);
             break;

             case FACETEDGE: 
              id = first = get_edge_fe(parent);
              if ( !valid_element(id) )
              { sprintf(errmsg,"Edge %s has no facets.\n",ELNAME(parent));
                sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                     file_names[node->file_no],node->line_no);
                kb_error(1281,errmsg,RECOVERABLE);
              }
              for ( n = 1 ; n <= ord ; n++ )
              { id = get_next_facet(id);
                if ( equal_id(id,first) )
                { sprintf(errmsg,
                    "Facetedge index, %d, exceeds valence on edge %s.\n",
                       ord+1,ELNAME(parent));
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                       file_names[node->file_no],node->line_no);
                  kb_error(1282,errmsg,RECOVERABLE);
                }
              }
              next_id = id;
              break;
            }
            break;
          
         case FACET:
             switch ( node->op1.eltype /* subtype */ )
             { case VERTEX:
                 if ( (web.representation == SIMPLEX) ||
                      (web.modeltype == LAGRANGE) )
                 { if ( ord >= web.skel[FACET].ctrlpts )
                   { sprintf(errmsg,
                  "Vertex index, %d, exceeds number of vertices on facet %s.\n",
                        ord+1,ELNAME(parent));
                    sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                     kb_error(3503,errmsg,RECOVERABLE);
                   }
                   next_id = get_facet_vertices(parent)[ord];
                   break;
                 }
                 if ( web.representation == SOAPFILM )
                 { if ( ord >= web.skel[FACET].ctrlpts )
                   { sprintf(errmsg,
                  "Vertex index, %d, exceeds number of vertices on facet %s.\n",
                        ord+1,ELNAME(parent));
                    sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                     kb_error(3504,errmsg,RECOVERABLE);
                   }
                   fe = get_facet_fe(parent);
                   if ( ord < 3 )
                   { for ( i = 0 ; i < ord ; i++ )
                       fe = get_next_edge(fe);
                     next_id = get_fe_tailv(fe);
                   }
                   else 
                   { for ( i = 3 ; i < ord ; i++ )
                       fe = get_next_edge(fe);
                     next_id = get_fe_midv(fe);
                   }
                   break;
                 }

                 /* now string */
                 id = first = get_facet_fe(parent);
                 if ( !valid_element(id) )
                 {  sprintf(errmsg,"Facet has no edges.\n");
                    sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                         file_names[node->file_no],node->line_no);
                    kb_error(1283,errmsg,RECOVERABLE);
                 }
                 if ( inverted(parent) ) 
                 { facetedge_id nextfe=id,startfe=id;
                   /* go back round to find start */
                   do
                   { id = nextfe;
                     nextfe = get_prev_edge(id);
                   } while ( valid_id(nextfe) && !equal_id(nextfe,startfe) );
                   first = id;
                 }
                 for ( n = 1 ; n <= ord ; n++ )
                   { edge_id next_edge = get_next_edge(id);
                     if ( !valid_id(next_edge) && n == ord )
                     { next_id = get_fe_headv(id);
                       id = NULLID;
                       break;
                     } 
                     id = next_edge;
                     if ( !valid_id(id) || equal_id(id,first) )
                     { sprintf(errmsg,
                    "Vertex index, %d, exceeds number of vertices on facet %s.\n",
                          ord+1,ELNAME(parent));
                      sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                       kb_error(1284,errmsg,RECOVERABLE);
                     }
                   }
                 if ( valid_id(id) ) 
                    next_id = get_fe_tailv(id);

                break;

               case EDGE:
                 id = first = get_facet_fe(parent);
                 if ( !valid_element(id) )
                 { sprintf(errmsg,"Facet has no edges.\n");
                   sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                        file_names[node->file_no],node->line_no);
                   kb_error(1285,errmsg,RECOVERABLE);
                 }

                 if ( inverted(parent) && (web.representation == STRING) ) 
                 { facetedge_id nextfe=id,startfe=id;
                   /* go back round to find start */
                   do
                   { id = nextfe;
                     nextfe = get_prev_edge(id);
                   } while ( valid_id(nextfe) && !equal_id(nextfe,startfe) );
                   first = id;
                 }
                 for ( n = 1 ; n <= ord ; n++ )
                   { id = get_next_edge(id);
                     if ( !valid_id(id) || equal_id(id,first) )
                     { sprintf(errmsg,
                    "Edge index, %d, exceeds number of edges on facet %s.\n",
                          ord+1,ELNAME(parent));
                      sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                          file_names[node->file_no],node->line_no);
                       kb_error(1286,errmsg,RECOVERABLE);
                     }
                   }
                   next_id = get_fe_edge(id);
                 break;
             
                case BODY:
                  switch ( ord )
                  { case 0: next_id = get_facet_body(parent); break;
                    case 1: next_id = get_facet_body(inverse_id(parent)); 
                            break;
                    default: 
                      sprintf(errmsg,
                       "Illegal facet body index %d; must be 1 or 2.\n",ord+1);
                      sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                       file_names[node->file_no],node->line_no);
                      kb_error(1287,errmsg, RECOVERABLE);
                      break;
                 }
                 if ( !valid_id(next_id) )
                 { sprintf(errmsg,"Facet %d does not have body of index %d.\n",
                      oid(parent),ord+1);
                   sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                     file_names[node->file_no],node->line_no);
                   kb_error(2070,errmsg,RECOVERABLE);
                 }
                 break;
             } 
             break;
         case BODY:
             switch ( node->op1.eltype /* subtype */ )
             { case FACET:

             id = first = get_body_facet(parent);
             for ( n = 1 ; n <= ord ; n++ )
              {
                id = get_next_body_facet(id);
                if ( equal_id(id,first) )
                { id = NULLID; 
                  break;
                }
               }
               if ( !valid_id(id) )
                { sprintf(errmsg,
                   "Facet index, %d, exceeds number of facets on body %s.\n",
                         ord+1, ELNAME(parent));
                  sprintf(errmsg+strlen(errmsg),"(source file %s, line %d)\n",
                     file_names[node->file_no],node->line_no);
                        kb_error(1288,errmsg,RECOVERABLE);
                }
              
             next_id = id;
            break;
           }
           break;

         case FACETEDGE:
             switch ( node->op1.eltype /* subtype */ )
             { case EDGE: next_id = get_fe_edge(parent); break;
               case FACET: next_id = get_fe_facet(parent); break;
             }
             break;
          }

          return next_id;

       } // end get_indexed_subtype()
