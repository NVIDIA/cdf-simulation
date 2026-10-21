%{
/********************************************************************
*        
*  File:  datafile.lex, lexyy.c  
*
*  Contents: lexical analyzer for evolver data files.
*            Constructed from datafile.lex by lex.
*/ 

/* will have my own input() and output() */

#ifdef FLEX_SCANNER
#define YY_INPUT(buf,result,max_size) \
     { \
       int c = kb_input(); \
       result = (c == 0) ? YY_NULL : (buf[0] = c, 1); \
     }
#else
/* lex scanner */
#undef input
#undef unput
#define input kb_input
#define unput rawunput
#endif

#include "include.h"
#include "lex.h"
#include "ytab.h"

static int previous_char = 0;  /* for CR conversion */
static int BUFFSIZE = 0; /* size of lex buffer */
static char *buff = NULL; /* lex buffer, expandable */
static int  spot = 0;
#define ERRBUFFSIZE 80
static char errbuff[ERRBUFFSIZE+4];  /* for reporting spot of error */
static int  errspot;
extern int ubuff_spot;

struct ckey { char *name; int token; } const_expr_keywords[] =
{
  {"pi",PI_TOK},
  {"e",E_TOK},
  {"g",G_TOK}
};

void savein (int);
void get_more_input(void);
int keyword_compare ( struct ckey *, struct ckey *);

/* Wrapper for yylex() to buffer look-ahead token */
#define UNPUTMAX 10
int unputted_tok[UNPUTMAX];  /* unput token number   */
int unput_tok_count;  /* 1 if token was unput */
YYSTYPE yylval;

char kb_input_new (void);
void kb_unput(char);
int kblex (void);
#define KB_INPUT kb_input_new
#define KB_UNPUT kb_unput

/* for generation of bad tokens for error message generation */
/* for use with -E command line option */
int next_err_token = 0;  /* token in file for which to generate error */
int token_count; /* resets on file initialization */
int err_tok_gen_flag;  /* whether -E option in effect */

/***********************************************************************
* function: kb_yylex()
* purpose: to be called to get next token.
*/
int kb_yylex(YYSTYPE *lvalp  /* for pure parser destination for yylval */)
{
  if ( lvalp )
  { PROF_FINISH(yyparse); /* exclude from yyparse time */
  }

  PROF_START(yylex);

  if ( unput_tok_count ) 
  { tok = unputted_tok[--unput_tok_count]; 
  }
  else 
  { 
    /* clean out previous data */
    /* memset(&yylval,0,sizeof(yylval)); time hog*/
    /*  tok = yylex(); */
    if ( err_tok_gen_flag )
    { if ( token_count == next_err_token )
      { tok = GEN_ERROR_TOKEN;
        if ( yytext )
          strcpy(yytext,"generated error");
        next_err_token++;
        token_count += 10000000; /* just one error per file */
        return tok;
      }
      token_count++;
    }
    tok = kblex();
  }

  switch ( tok )
  { case '(' : parens++; break;
    case ')' : parens--; break;
    case '{' : brace_depth++; break;
    case '}' : brace_depth--; break;
  }

  if ( lvalp ) 
  {
    {
      /* non-movsd substitute for     *lvalp = yylval; */
      int i;
      int *dest = (int*)lvalp;
      int *src  = (int*)&yylval;
      for ( i = 0 ; i < sizeof(yylval)/sizeof(int) ; i++ )
         *(dest++) = *(src++);
    }

    PROF_START(yyparse);
  }


  PROF_FINISH(yylex);
  return tok;
} // end kb_yylex()

/*********************************************************************
* function: unput_tok()
* purpose: For unputting unneeded look-ahead token 
*/
void unput_tok()
{ 
  if ( unput_tok_count >= UNPUTMAX )
    kb_error(2324,"Internal error: unputted_tok stack overflow.\n",DATAFILE_ERROR);
  unputted_tok[unput_tok_count++] = tok;
  /* fprintf(stderr,"UNPUT %3d %s %s\n",tok,tokname(tok),yytext); */
  /* tok = UNPUTTED_TOK; */ /* some places depend on knowing unput lookahead */
 
} // end unput_tok()

int rawinput(void);
int yyback(int *,int);
/* #define yyoutput yyout_unused */
/* #define yyunput yyunput_unused */

extern FILE *data_fd;
extern char *cmdptr;
int line_no = 1;

int macro_flag;  /* tells macro() identifier is being #define'd */
/* return values for macro() */
#define NO_MACRO  0
#define WAS_MACRO_DEF 1
#define MACRO_EXPANDED 2

char *whitespace = " \t\r,:;";   /* whitespace */

#define SUBMAX 500
int  macro_max;
int  macro_subs_top;  /* index of top of string space */
int  macro_subs_max;   /* space allocated for strings */

struct dkey { char *name; int token; } datafile_keywords[] =
{ {"calculate_in_3d",CALC_IN_3D_TOK},
  {"vertices_predicted",VERTICES_PREDICTED_TOK},
  {"edges_predicted",EDGES_PREDICTED_TOK},
  {"facets_predicted",FACETS_PREDICTED_TOK},
  {"bodies_predicted",BODIES_PREDICTED_TOK},
  {"facetedges_predicted",FACETEDGES_PREDICTED_TOK},
  {"quantities_predicted",QUANTITIES_PREDICTED_TOK},
  {"method_instances_predicted",METHOD_INSTANCES_PREDICTED_TOK},
  {"opacity",OPACITY_TOK},
  {"mpi_local_bodies",MPI_LOCAL_BODIES_NODE},
  {"clip_coeff",CLIP_COEFF_TOK},
  {"slice_coeff",SLICE_COEFF_TOK},
  {"on_constraint",ON_CONSTRAINT_TOK},
  {"hit_constraint",HIT_CONSTRAINT_TOK},
  {"value_of_constraint",VALUE_OF_CONSTRAINT_TOK},
  {"on_boundary",ON_BOUNDARY_TOK},
  {"on_quantity",ON_QUANTITY_TOK},
  {"on_method_instance",ON_METHOD_INSTANCE_TOK},
  {"content_rank",CONTENT_RANK_TOK},
  {"high_constraint",V_HIGH_CONSTRAINT},
  {"high_boundary",V_HIGH_BOUNDARY},
  {"suppress_warning",SUPPRESS_WARNING_TOK},
  {"unsuppress_warning",UNSUPPRESS_WARNING_TOK},
  {"volume_method_name",VOLUME_METHOD_NAME_TOK},
  {"partner_hitting",PARTNER_HITTING_TOK},
  {"procedure",PROCEDURE_WORD_TOK},
  {"display_origin",DISPLAY_ORIGIN_TOK},
  {"length_method_name",LENGTH_METHOD_NAME_TOK},
  {"area_method_name",AREA_METHOD_NAME_TOK},
  {"hessian_special_normal_vector",HESSIAN_SPECIAL_NORMAL_VECTOR_TOK},
  {"keep_originals",KEEP_ORIGINALS_TOK},
  {"element_modulus",ELEMENT_MODULUS_TOK},
  {"swap_colors",SWAP_COLORS_TOK},
  {"self",SELF_TOK},
  {"conserved",CONSERVED_TOK},
  {"actual_volume",ACTUAL_VOLUME_TOK},
  {"keep_macros",KEEP_MACROS_TOK},
  {"pdelta",PDELTA_TOK},
  {"on_assign_call",ON_ASSIGN_CALL_TOK},
  {"tolerance",TOLERANCE_TOK},
  {"lagrange_multiplier",LAGRANGE_MULTIPLIER_TOK},
  {"evolver_version",VERSION_TOK},
  {"orientation",ORIENTATION_TOK},
  {"ignore_constraints",IGNORE_CONSTRAINTS_TOK},
  {"ignore_fixed",IGNORE_FIXED_TOK},
  {"load_library",LOAD_LIBRARY_TOK},
  {"interp_bdry_param",INTERP_BDRY_PARAM_TOK},
  {"axial_point",AXIAL_POINT_TOK},
  {"lagrange",LAGRANGE_TOK},
  {"lagrange_order",LAGRANGE_ORDER_TOK},
  {"parameter_1",PARAMETER_1_TOK},
  {"parameter_2",PARAMETER_2_TOK},
  {"optimizing_parameter",OPTIMIZING_PARAMETER_TOK},
  {"optimising_parameter",OPTIMIZING_PARAMETER_TOK},
  {"everything_quantities",V_EVERYTHING_QUANTITIES_TOK},
  {"value",VALUE_TOK},
  {"target",TARGET_TOK},
  {"define",DEFINE_TOK},
  {"attribute",ATTRIBUTE_TOK},
  {"method_instance",METHOD_INSTANCE_TOK},
  {"method",METHOD_TOK},
  {"scalar_integrand",SCALAR_INTEGRAND_TOK},
  {"vector_integrand",VECTOR_INTEGRAND_TOK},
  {"k_vector_order",K_VEC_ORDER_TOK},
  {"form_integrand",FORM_INTEGRAND_TOK},
  {"mobility",MOBILITY_TOK},
  {"mobility_tensor",MOBILITY_TENSOR_TOK},
  {"bare",BARE_TOK},
  {"boundary_curvature",BOUNDARY_CURVATURE_TOK},
  {"modulus",MODULUS_TOK},
  {"info_only",INFO_ONLY_TOK},
  {"global_method",GLOBAL_METHOD_TOK},
  {"global",GLOBAL_TOK},
  {"area_fixed",AREA_FIXED_TOK},
  {"fixed_area",AREA_FIXED_TOK},
  {"view_matrix",VIEW_MATRIX_TOK},
  {"view_transforms",VIEW_TRANSFORMS_TOK},
  {"view_transform_generators",VIEW_TRANSFORM_GENS_TOK},
  {"homothety",HOMOTHETY_TOK},
  {"approximate_curvature",APPROX_CURV_TOK},
  {"approx_curvature",APPROX_CURV_TOK},
  {"phasefile",PHASEFILE_TOK},
  {"phase",PHASE_TOK},
  {"autopop",AUTOPOP_TOK},
  {"autopop_quartic",AUTOPOP_QUARTIC_TOK},
  {"autochop",AUTOCHOP_TOK},
  {"total_time",TOTAL_TIME_TOK},
  {"effective_area",EFFECTIVE_AREA_TOK},
  {"runge_kutta",RUNGE_KUTTA_TOK},
  {"color",COLOR_TOK},
  {"backcolor",BACKCOLOR_TOK},
  {"frontcolor",FRONTCOLOR_TOK},
  {"mean_curvature_integral",MEAN_CURV_INT_TOK},
  {"normal_curvature",NORMAL_CURVATURE_TOK},
  {"square_curvature",SQUARE_CURVATURE_TOK},
  {"squared_curvature",SQUARE_CURVATURE_TOK},
  {"square_gaussian_curvature",SQGAUSS_TOK},
  {"squared_gaussian_curvature",SQGAUSS_TOK},
  {"gauss_curvature",GAUSS_CURVATURE_TOK},
  {"insulating_knot_energy",INSULATING_KNOT_ENERGY_TOK},
  {"conducting_knot_energy",CONDUCTING_KNOT_ENERGY_TOK},
  {"space_dimension",SPACE_DIMENSION_TOK},
  {"surface_dimension",SURFACE_DIMENSION_TOK},
  {"simplex_representation",SIMPLEX_REP_TOK},
  {"metric",METRIC_TOK},
  {"klein_metric",KLEIN_METRIC_TOK},
  {"conformal_metric",CONFORMAL_TOK},
  {"fixed",FIXED_TOK},
  {"no_refine",NO_REFINE_TOK},
  {"no_transform",NO_TRANSFORM_TOK},
  {"hit_partner",HIT_PARTNER_TOK},
  {"centerofmass",CENTEROFMASS_TOK},
  {"no_display",NODISPLAY_TOK},
  {"noncontent",NONCONTENT_TOK},
  {"efixed",EFIXED_TOK},
  {"symmetry_group",SYMMETRY_GROUP_TOK},
  {"wrap",WRAP_TOK},
  {"torus",TORUS_TOK},
  {"torus_filled",TORUS_FILLED_TOK},
  {"torus_periods",TORUS_PERIODS_TOK},
  {"periods",PERIODS_TOK},
  {"display_periods",DISPLAY_PERIODS_TOK},
  {"string",STRING_TOK},
  {"soapfilm",SOAPFILM_TOK},
  {"wulff",WULFF_TOK},
  {"boundary",BOUNDARY_TOK},
  {"boundaries",BOUNDARY_TOK},
  {"constraint",CONSTRAINT_TOK},
  {"constraints",CONSTRAINT_TOK},
  {"surface_energy",SURFACE_ENERGY_TOK},
  {"formula",FUNCTION_TOK},
  {"function",FUNCTION_TOK},
  {"parameter",PARAMETERS_TOK},
  {"parameters",PARAMETERS_TOK},
  {"parameter_file",PARAMETER_FILE_TOK},
  {"symmetric_content",SYMMETRIC_CONTENT_TOK},
  {"integral_order",V_INTEGRAL_ORDER},
  {"integral_order_1d",V_INTEGRAL_ORDER_1D},
  {"integral_order_2d",V_INTEGRAL_ORDER_2D},
  {"integration_order",V_INTEGRAL_ORDER},
  {"integration_order_1d",V_INTEGRAL_ORDER_1D},
  {"integration_order_2d",V_INTEGRAL_ORDER_2D},
  {"constraint_tolerance",CONSTRAINT_TOLERANCE_TOK},
  {"convex",CONVEX_TOK},
  {"nonwall",NONWALL_TOK},
  {"nonnegative",NONNEGATIVE_TOK},
  {"nonpositive",NONPOSITIVE_TOK},
  {"global",GLOBAL_TOK},
  {"energy",ENERGY_TOK},
  {"content",CONTENT_TOK},
  {"quadratic",QUADRATIC_TOK},
  {"linear",LINEAR_TOK},
  {"area_normalization",MEAN_CURV_TOK},
  {"jiggle",JIGGLE_TOK},
  {"diffusion",DIFFUSION_TOK},
  {"merit_factor",MERITFACTOR_TOK},
  {"gravity_constant",GRAV_CONST_TOK},
  {"spring_constant",SPRING_CONSTANT_TOK},
  {"gap_constant",GAP_CONSTANT_TOK},
  {"scale",SCALE_TOK},
  {"pscale",PSCALE_TOK},
  {"temperature",TEMPERATURE_TOK},
  {"pressure",PRESSURE_TOK},
  {"volume",VOLUME_TOK},
  {"density",DENSITY_TOK},
  {"tension",DENSITY_TOK},
  {"nodisplay",NODISPLAY_TOK},
  {"scale_limit",SCALE_LIMIT_TOK},
  {"zoom_vertex",ZOOM_VERTEX_TOK},
  {"zoom_radius",ZOOM_RADIUS_TOK},
  {"quantity",QUANTITY_TOK},
  {"volconst",VOLCONST_TOK},
  {"read",READ_TOK},
  {"and",AND_TOK},
  {"or",OR_TOK},
  {"not",NOT_TOK},
  {"mod",'%'},
  {"imod",IMOD_TOK},
  {"idiv",IDIV_TOK},
  {"dot_product",DOT_TOK},
  {"pi",PI_TOK},
  {"e",E_TOK},
  {"g",G_TOK},
  {"original",ORIGINAL_TOK},
  {"vertices",VERTICES_TOK},
  {"vertex",VERTICES_TOK},
  {"edges",EDGES_TOK},
  {"edge",EDGES_TOK},
  {"faces",FACES_TOK},
  {"face",FACES_TOK},
  {"facets",FACES_TOK},
  {"facet",FACES_TOK},
  {"bodies",BODIES_TOK},
  {"body",BODIES_TOK},
  {"facet_edges",FACETEDGES_TOK},
  {"facet_edge",FACETEDGES_TOK},
  {"facetedges",FACETEDGES_TOK},
  {"facetedge",FACETEDGES_TOK}
};


struct ckey mathfunc_keywords[] = {
  {"wrap_inverse",WRAP_INVERSE_NODE},
  {"ellipticK",ELLIPTICK_NODE},
  {"ellipticE",ELLIPTICE_NODE},
  {"log",LOG_NODE},
  {"exp",EXP_NODE},
  {"sin",SIN_NODE},
  {"cos",COS_NODE},
  {"tan",TAN_NODE},
  {"asin",ASIN_NODE},
  {"acos",ACOS_NODE},
  {"atan",ATAN_NODE},
  {"sinh",SINH_NODE},
  {"cosh",COSH_NODE},
  {"tanh",TANH_NODE},
  {"asinh",ASINH_NODE},
  {"acosh",ACOSH_NODE},
  {"atanh",ATANH_NODE},
  {"sqrt",SQRT_NODE},
  {"sqr",SQR_NODE},
  {"ceil",CEIL_NODE},
  {"floor",FLOOR_NODE},
  {"abs",ABS_NODE}
};

struct ckey mathfunc2_keywords[] = { /* two arguments */
  {"wrap_compose",WRAP_COMPOSE_NODE},
  {"atan2",ATAN2_NODE},
  {"incompleteEllipticF",INCOMPLETE_ELLIPTICF_NODE},
  {"incompleteEllipticE",INCOMPLETE_ELLIPTICE_NODE},
  {"maximum",MAXIMUM_NODE},
  {"minimum",MINIMUM_NODE},
  {"pow",POW_NODE}
};

struct ckey command_keywords[] =
{ {"normal",NORMAL_TOK},
  {"nonnegative",NONNEGATIVE_TOK},
  {"nonpositive",NONPOSITIVE_TOK},
  {"normal_vector",NORMAL_VECTOR_TOK},
  {"no_dump",NO_DUMP_TOK},
  {"p_velocity",P_VELOCITY_TOK},
  {"p_force",P_FORCE_TOK},
  {"no_hessian_normal",NO_HESSIAN_NORMAL_TOK},
  {"make_thread_lists",MAKE_THREAD_LISTS_TOK},
  {"facet_crosscut",FACET_CROSSCUT_TOK},
  {"detorus",DETORUS_TOK},
  {"is_constraint",IS_CONSTRAINT_TOK},
  {"valid_constraint",VALID_CONSTRAINT_TOK},
  {"valid_boundary",VALID_BOUNDARY_TOK},
  {"profiling",PROFILING_TOK},
  {"reset_profiling",RESET_PROFILING_TOK},
  {"suppress_warning",SUPPRESS_WARNING_TOK},
  {"unsuppress_warning",UNSUPPRESS_WARNING_TOK},
  {"delete_text",DELETE_TEXT_TOK},
  {"display_text", DISPLAY_TEXT_TOK},
  {"simplex_to_fe", SIMPLEX_TO_FE_TOK},
  {"addload", ADDLOAD_TOK},
  {"replace_load", REPLACE_LOAD_TOK},
  {"whereami",WHEREAMI_TOK},
  {"breakpoint",BREAKPOINT_TOK},
  {"breakpoints",BREAKPOINT_TOK},
  {"abort",ABORT_TOK},
  {"subcommand",SUBCOMMAND_TOK},
  {"global",GLOBAL_TOK},
  {"repartition",REPARTITION_TOK},
  {"free_discards",FREE_DISCARDS_TOK},
  {"dump_memlist",DUMP_MEMLIST_TOK},
  {"reverse_orientation",REVERSE_ORIENTATION_TOK},
  {"matrix_multiply",MATRIX_MULTIPLY_TOK},
  {"matrix_inverse",MATRIX_INVERSE_TOK},
  {"matrix_determinant",MATRIX_DETERMINANT_TOK},
  {"mid_edge",MID_EDGE_TOK},
  {"mid_facet",MID_FACET_TOK},
  {"flush_counts",FLUSH_COUNTS_TOK},
  {"valid_element",VALID_ELEMENT_TOK},
  {"reset_counts",RESET_COUNTS_TOK},
  {"vertex_merge",MERGE_VERTEX_TOK},
  {"edge_merge",MERGE_EDGE_TOK},
  {"facet_merge",MERGE_FACET_TOK},
  {"mpi_task",MPI_TASK_ATTR_TOK},
  {"mean_curvature",MEAN_CURVATURE_TOK},
  {"element_modulus",ELEMENT_MODULUS_TOK},
  {"ignore_constraints",IGNORE_CONSTRAINTS_TOK},
  {"ignore_fixed",IGNORE_FIXED_TOK},
  {"global_method",GLOBAL_METHOD_TOK},
  {"scalar_integrand",SCALAR_INTEGRAND_TOK},
  {"vector_integrand",VECTOR_INTEGRAND_TOK},
  {"k_vector_order",K_VEC_ORDER_TOK},
  {"form_integrand",FORM_INTEGRAND_TOK},
  {"info_only",INFO_ONLY_TOK},
  {"method",METHOD_TOK},
  {"parallel_exec",PARALLEL_EXEC_TOK},
  {"task_exec",TASK_EXEC_TOK},
  {"pop",POP_TOK},
  {"pop_tri_to_edge",POP_TRI_TO_EDGE_TOK},
  {"pop_edge_to_tri",POP_EDGE_TO_TRI_TOK},
  {"pop_quad_to_quad",POP_QUAD_TO_QUAD_TOK},
  {"local",LOCAL_TOK},
  {"for",FOR_TOK},
  {"warning_messages",WARNING_MESSAGES_TOK},
  {"equiangulate",EQUIANGULATE_TOK},
  {"no_refine",NO_REFINE_TOK},
  {"no_transform",NO_TRANSFORM_TOK},
  {"hit_partner",HIT_PARTNER_TOK},
  {"centerofmass",CENTEROFMASS_TOK},
  {"no_display",NODISPLAY_TOK},
  {"vertexnormal",VERTEXNORMAL_TOK},
  {"colorfile",COLORFILE_TOK},
  {"date_and_time",DATE_AND_TIME_TOK},
  {"evolver_version",EVOLVER_VERSION_TOK},
  {"exprint",EXPRINT_TOK},
  {"energy",ENERGY_TOK},
  {"info_only",INFO_ONLY_TOK},
  {"conserved",CONSERVED_TOK},
  {"exec",EXEC_TOK},
  {"wrap_vertex",WRAP_VERTEX_TOK},
  {"self",SELF_TOK},
  {"is_defined",IS_DEFINED_TOK},
  {"nodisplay",NODISPLAY_TOK},
  {"no_display",NODISPLAY_TOK},
  {"function",FUNCTION_TOK},
  {"reorder_storage",REORDER_STORAGE_TOK},
  {"renumber_all",RENUMBER_ALL_TOK},
/*  {"view_matrix",VIEW_MATRIX_TOK}, */
  {"pause",PAUSE_TOK},
  {"pdelta",PDELTA_TOK},
  {"on_assign_call",ON_ASSIGN_CALL_TOK},
  {"pscale",PSCALE_TOK},
  {"scale",SCALE_TOK},
  {"tolerance",TOLERANCE_TOK},
  {"method_instance",METHOD_INSTANCE_TOK},
  {"logfile",LOGFILE_TOK},
  {"keylogfile",KEYLOGFILE_TOK},
  {"frontbody",FRONTBODY_TOK},
  {"backbody",BACKBODY_TOK},
  {"new_vertex",NEWVERTEX_TOK},
  {"new_edge",NEWEDGE_TOK},
  {"new_facet",NEWFACET_TOK},
  {"new_body",NEWBODY_TOK},
  {"noncontent",NONCONTENT_TOK},
/*  {"inverse_periods",INVERSE_PERIODS_TOK}, */
  {"return",RETURN_TOK},
  {"postscript",POSTSCRIPT_TOK},
  {"ooglfile",OOGLFILE_TOK},
  {"binary_off_file",BINARY_OFF_FILE_TOK},
  {"vertex_average",VERTEX_AVERAGE_TOK},
  {"raw_vertex_average",RAW_VERTEX_AVERAGE_TOK},
  {"rawest_vertex_average",RAWEST_VERTEX_AVERAGE_TOK},
  {"axial_point",AXIAL_POINT_TOK},
  {"metis_factor",METIS_FACTOR_TOK},
  {"lagrange",LAGRANGE_TOK},
  {"metis",METIS_TOK},
  {"metis_readjust",METIS_READJUST_TOK},
  {"body_metis",BODY_METIS_TOK},
  {"kmetis",KMETIS_TOK},
  {"ometis",OMETIS_TOK},
  {"sizeof",SIZEOF_TOK},
  {"move",MOVE_TOK},
  {"geompipe",GEOMPIPE_TOK},
  {"convert_to_quantities",CONVERT_TO_QUANTS_TOK},
  {"edgeswap",EDGESWAP_TOK},
  {"t1_edgeswap",T1_EDGESWAP_TOK},
/*  {"torus_periods",TORUS_PERIODS_TOK}, */
  {"break",BREAK_TOK},
  {"volfixed",FIXEDVOL_TOK},
  {"continue",CONTINUE_TOK},
  {"volconst",VOLCONST_TOK},
  {"ritz",RITZ_TOK},
  {"orientation",ORIENTATION_TOK},
  {"eigenprobe",EIGENPROBE_TOK},
  {"lanczos",LANCZOS_TOK},
  {"tetra_point",TETRA_POINT_TOK},
  {"triple_point",TRIPLE_POINT_TOK},
  {"value",VALUE_TOK},
  {"target",TARGET_TOK},
  {"datafilename",DATAFILENAME_TOK},
  {"geomview",GEOMVIEW_TOK},
  {"saddle",HESSIAN_SADDLE_TOK},
  {"midv",MIDV_TOK},
  {"define",DEFINE_TOK},
  {"attribute",ATTRIBUTE_TOK},
  {"attributes",ATTRIBUTE_TOK},
  {"string",STRING_TOK},
  {"sobolev",SOBOLEV_TOK},
  {"sobolev_seek",SOBOLEV_SEEK_TOK},
  {"dirichlet",DIRICHLET_TOK},
  {"dirichlet_seek",DIRICHLET_SEEK_TOK},
  {"wrap",WRAP_TOK},
  {"total",TOTAL_TOK},
  {"history",HISTORY_TOK},
  {"fix",FIX_TOK},
  {"unfix",UNFIX_TOK},
  {"bare",BARE_TOK},
  {"phase",PHASE_TOK},
  {"foreach",FOREACH_TOK},
  {"rebody", REBODY_TOK},
  {"burchard", BURCHARD_TOK},
  {"close_show",CLOSE_SHOW_TOK},
  {"show_off",CLOSE_SHOW_TOK},
/*  {"view_transforms",VIEW_TRANSFORMS_TOK}, */
/*  {"view_transform_swap_colors",VIEW_TRANSFORM_SWAP_COLORS_TOK}, */
/*  {"view_transform_parity",VIEW_TRANSFORM_PARITY_TOK}, */
  {"transform_depth",TRANSFORM_DEPTH_TOK},
  {"transform_expr",TRANSFORM_EXPR_TOK},
  {"modulus",MODULUS_TOK},
  {"boundary",BOUNDARY_TOK},
  {"on_constraint",ON_CONSTRAINT_TOK},
  {"hit_constraint",HIT_CONSTRAINT_TOK},
  {"value_of_constraint",VALUE_OF_CONSTRAINT_TOK},
  {"on_boundary",ON_BOUNDARY_TOK},
  {"on_quantity",ON_QUANTITY_TOK},
  {"on_method_instance",ON_METHOD_INSTANCE_TOK},
  {"hessian",HESSIAN_TOK},
  {"hessian_seek",HESSIAN_SEEK_TOK},
  {"hessian_menu",HESSIAN_MENU_TOK},
  {"help",HELP_TOK},
  {"quit",QUIT_TOK},
  {"exit",QUIT_TOK},
  {"bye",QUIT_TOK},
  {"dump",DUMP_TOK},
  {"load",LOAD_TOK},
  {"permload",PERMLOAD_TOK},
  {"quantity",QUANTITY_TOK},
  {"spring_constant",GAP_CONSTANT_TOK},
  {"gap_constant",GAP_CONSTANT_TOK},
  {"notch",NOTCH_TOK},
  {"while",WHILE_TOK},
  {"do",DO_TOK},
  {"if",IF_TOK},
  {"then",THEN_TOK},
  {"else",ELSE_TOK},
  {"histogram",HISTOGRAM_TOK},
  {"loghistogram",LOGHISTOGRAM_TOK},
  {"show_expr",SHOW_EXPR_TOK},
  {"show_trans",SHOW_TRANS_TOK},
  {"dihedral",DIHEDRAL_TOK},
  {"max",MAX_TOK},
  {"min",MIN_TOK},
  {"avg",AVG_TOK},
  {"sum",SUM_TOK},
  {"count",COUNT_TOK},
  {"print",PRINT_TOK},
  {"eprint",EPRINT_TOK},
  {"printf",PRINTF_TOK},
  {"errprintf",ERRPRINTF_TOK},
  {"sprintf",SPRINTF_TOK},
  {"binary_printf",BINARY_PRINTF_TOK},
  {"procedures",PROCEDURES_TOK},
  {"procedure",PROCEDURE_WORD_TOK},
  {"on",ON_TOK},
  {"counts",COUNTS_TOK},
  {"extrapolate",EXTRAPOLATE_TOK},
  {"off",OFF_TOK},
  {"areaweed",AREAWEED_TOK},
  {"edgeweed",EDGEWEED_TOK},
  {"edge_divide",EDGEDIVIDE_TOK},
  {"alice",ALICE_TOK},
  {"stability_test",STABILITY_TEST_TOK},
  {"zoom",ZOOM_TOK},
  {"utest",UTEST_TOK},
  {"system",SYSTEM_TOK},
  {"chdir",CHDIR_TOK},
  {"longj",LONG_JIGGLE_TOK},
  {"rawv",RAW_VERAVG_TOK},
  {"rawestv",RAWEST_VERAVG_TOK},
  {"go",GO_TOK},
  {"refine",REFINE_TOK},
  {"check",CHECK_TOK},
  {"show_vol",SHOW_VOL_TOK},
  {"id",ID_TOK},
  {"oid",OID_TOK},
  {"and",AND_TOK},
  {"or",OR_TOK},
  {"not",NOT_TOK},
  {"mod",'%'},
  {"imod",IMOD_TOK},
  {"idiv",IDIV_TOK},
  {"dot_product",DOT_TOK},
  {"pi",PI_TOK},
  {"e",E_TOK},
  {"g",G_TOK},
  {"original",ORIGINAL_TOK},
  {"fixed",FIXED_TOK},
  {"vertices",VERTICES_TOK},
  {"vertex",VERTICES_TOK},
  {"edges",EDGES_TOK},
  {"edge",EDGES_TOK},
  {"facets",FACETS_TOK},
  {"facet",FACETS_TOK},
  {"faces",FACETS_TOK},
  {"face",FACETS_TOK},
  {"facet_edges",FACETEDGES_TOK},
  {"facet_edge",FACETEDGES_TOK},
  {"facetedges",FACETEDGES_TOK},
  {"facetedge",FACETEDGES_TOK},
  {"bodies",BODIES_TOK},
  {"body",BODIES_TOK},
  {"topinfo",TOPINFO_TOK},
  {"bottominfo",BOTTOMINFO_TOK},
  {"length",LENGTH_TOK},
  {"valence",VALENCE_TOK},
  {"area",AREA_TOK},
  {"volume",VOLUME_TOK},
  {"sqcurve",SQ_MEAN_CURV_TOK},
  {"where",WHERE_TOK},
  {"list",LIST_TOK},
  {"show",SHOW_TOK},
  {"showq",SHOWQ_TOK},
  {"delete",DELETE_TOK},
  {"dissolve",DISSOLVE_TOK},
  {"refine",REFINE_TOK},
  {"shell",SHELL_TOK},
  {"constraint",CONSTRAINT_TOK},
  {"pressure",PRESSURE_TOK},
  {"color",COLOR_TOK},
  {"backcolor",BACKCOLOR_TOK},
  {"frontcolor",FRONTCOLOR_TOK},
  {"opacity",OPACITY_TOK},
  {"volume",VOLUME_TOK},
  {"density",DENSITY_TOK},
  {"tension",DENSITY_TOK},
  {"set",SET_TOK},
  {"read",READ_TOK},
  {"unset",UNSET_TOK},
  {"recalc",RECALC_TOK},
  {"optimize",OPTIMIZE_TOK},
  {"optimise",OPTIMIZE_TOK},
  {"autochop",AUTOCHOP_TOK}
 }; 

struct ckey togglenames[] = {
  {"force_edgeswap",FORCE_EDGESWAP_NODE},
  {"show_all_edges",SHOW_ALL_EDGES_NODE},
  {"K_altitude_mode",K_ALTITUDE_FLAG_NODE},
  {"show_bounding_box",BOX_FLAG_NODE},
  {"septum_flag",SEPTUM_FLAG_NODE},
  {"view_transforms_use_unique_point",VIEW_TRANSFORMS_USE_UNIQUE_NODE},
  {"detorus_sticky",DETORUS_STICKY_NODE},
  {"immediate_autopop",IMMEDIATE_AUTOPOP_NODE},
  {"star_finagling",STAR_FINAGLING_NODE},
  {"force_deletion",FORCE_DELETION_NODE},
  {"function_quantity_sparse",FUNCTION_QUANTITY_SPARSE_NODE},
  {"slice_view",SLICE_VIEW_NODE},
  {"clip_view",CLIP_VIEW_NODE},
  {"quietload",QUIETLOAD_NODE},
  {"big_endian",BIG_ENDIAN_NODE},
  {"little_endian",LITTLE_ENDIAN_NODE},
  {"full_bounding_box",FULL_BOUNDING_BOX_NODE},
  {"pop_disjoin",POP_DISJOIN_NODE},
  {"pop_enjoin",POP_ENJOIN_NODE},
  {"pop_to_edge",POP_TO_EDGE_NODE},
  {"pop_to_face",POP_TO_FACE_NODE},
  {"mpi_debug",MPI_DEBUG_NODE},
  {"smooth_graph",SMOOTH_GRAPH_NODE},
  {"bezier_basis",BEZIER_BASIS_NODE},
  {"break_after_warning",BREAK_AFTER_WARNING_NODE},
  {"break_on_warning",BREAK_ON_WARNING_NODE},
  {"blas_flag",BLAS_FLAG_NODE},
  {"diffusion",DIFFUSION_NODE},
  {"augmented_hessian",AUGMENTED_HESSIAN_NODE},
  {"sparse_constraints",SPARSE_CONSTRAINTS_NODE},
  {"visibility_test", VISIBILITY_TEST_NODE},
  {"circular_arc_draw",CIRCULAR_ARC_DRAW_NODE},
  {"rgb_colors",RGB_COLORS_FLAG_NODE},
  {"kraynikpopvertex",KRAYNIKPOPVERTEX_FLAG_NODE},
  {"kraynikpopedge",KRAYNIKPOPEDGE_FLAG_NODE},
  {"sobolev_mode",SOBOLEV_MODE_NODE},
  {"dirichlet_mode",DIRICHLET_MODE_NODE},
  {"verbose",VERBOSE_NODE},
  {"torus_filled",TORUS_FILLED_NODE},
  {"ambient_pressure",AMBIENT_PRESSURE_NODE},
  {"backcull",BACKCULL_NODE},
  {"rotate_lights",ROTATE_LIGHTS_NODE},
  {"interp_normals",INTERP_NORMALS_NODE},
  {"volgrads_every",VOLGRADS_EVERY_NODE},
  {"zener_drag",ZENER_DRAG_NODE},
  {"hessian_special_normal",HESSIAN_SPECIAL_NORMAL_NODE},
  {"hessian_normal_perp",HESSIAN_NORMAL_PERP_NODE},
  {"show_all_quantities",SHOW_ALL_QUANTITIES_NODE},
  {"pscolorflag",PSCOLORFLAG_NODE},
  {"ps_colorflag",PSCOLORFLAG_NODE},
  {"ps_cmykflag",PS_CMYKFLAG_NODE},
  {"ps_gridflag",GRIDFLAG_NODE},
  {"gridflag",GRIDFLAG_NODE},
  {"crossingflag",CROSSINGFLAG_NODE},
  {"ps_crossingflag",CROSSINGFLAG_NODE},
  {"labelflag",LABELFLAG_NODE},
  {"ps_labelflag",LABELFLAG_NODE},
  {"hessian_normal_one",HESSIAN_NORMAL_ONE_NODE},
  {"interp_bdry_param",INTERP_BDRY_PARAM_NODE},
  {"hessian_double_normal",HESSIAN_DOUBLE_NORMAL_NODE},
  {"h_inverse_metric",H_INVERSE_METRIC_NODE},
  {"squared_gradient",SQUARED_GRADIENT_NODE},
  {"linear_metric",LINEAR_METRIC_NODE},
  {"metric_convert",METRIC_CONVERT_NODE},
  {"quantities_only",QUANTITIES_ONLY_NODE},
  {"ysmp",YSMP_NODE},
  {"mkl",MKL_NODE},
  {"bunch_kaufman",BUNCH_KAUFMAN_NODE},
  {"bunch_kauffman",BUNCH_KAUFMAN_NODE},
  {"hessian_normal",HESSIAN_NORMAL_NODE},
  {"jiggle",JIGGLE_TOGGLE_NODE},
  {"assume_oriented",ASSUME_ORIENTED_NODE},
  {"ribiere",RIBIERE_CG_NODE},
  {"area_normalization",MEAN_CURV_NODE},
  {"force_pos_def",FORCE_POS_DEF_NODE},
  {"autodisplay",AUTODISPLAY_NODE},
  {"lagrange",LAGRANGE_NODE},
  {"quadratic",QUADRATIC_NODE},
  {"linear",LINEAR_NODE},
  {"show_inner",SHOW_INNER_NODE},
  {"area_fixed",AREA_FIXED_TOK},
  {"fixed_area",AREA_FIXED_TOK},
  {"clipped",CLIPPED_CELLS_NODE},
  {"clipped_cells",CLIPPED_CELLS_NODE},
  {"raw_cells",RAW_CELLS_NODE},
  {"connected",CONNECTED_CELLS_NODE},
  {"connected_cells",CONNECTED_CELLS_NODE},
  {"show_outer",SHOW_OUTER_NODE},
  {"colormap",COLORMAP_NODE},
  {"thicken",THICKEN_NODE},
  {"hessian_diff",HESSIAN_DIFF_NODE},
  {"normal_motion",NORMAL_MOTION_NODE},
  {"runge_kutta",RUNGE_KUTTA_NODE},
  {"deturck",DETURCK_NODE},
  {"kusner",KUSNER_NODE},
  {"view_4d",VIEW_4D_NODE},
  {"conf_edge",CONF_EDGE_SQCURV_NODE},
  {"mean_curvature_integral",MEAN_CURV_INT_NODE},
  {"sqgauss",SQGAUSS_NODE},
  {"autopop",AUTOPOP_NODE},
  {"autopop_quartic",AUTOPOP_QUARTIC_NODE},
  {"old_area",OLD_AREA_NODE}, 
  {"approx_curv",APPROX_CURV_NODE},
  {"check_increase",CHECK_INCREASE_NODE},
  {"debug",DEBUG_NODE},
  {"memdebug",MEMDEBUG_NODE},
  {"itdebug",ITDEBUG_NODE},
  {"gravity",GRAVITY_NODE},
  {"effective_area",EFFECTIVE_AREA_NODE},
  {"estimate",ESTIMATE_NODE},
  {"post_project",POST_PROJECT_NODE},
  {"transforms",TRANSFORMS_NODE},
  {"quiet",QUIET_NODE},
  {"quietgo",QUIETGO_NODE},
  {"hessian_quiet",HESSIAN_QUIET_NODE},
  {"conj_grad",CONJ_GRAD_NODE},
  {"homothety",HOMOTHETY_NODE},
  {"facet_colors",FACET_COLORS_NODE},
  {"shading",SHADING_NODE},
  {"div_normal_curvature",DIV_NORMAL_CURVATURE_NODE},
  {"normal_curvature",NORMAL_CURVATURE_NODE},
  {"boundary_curvature",BOUNDARY_CURVATURE_NODE},
  {"self_similar",SELF_SIMILAR_NODE},
  {"gv_binary",GV_BINARY_NODE},
  {"metric_conversion",METRIC_CONVERSION_NODE},
  {"autorecalc",AUTORECALC_NODE},
  {"pinning",PINNING_NODE}
  };
struct ckey colornames[] = {
    {"clear",CLEAR},     /* transparent */
    {"transparent",CLEAR},     /* transparent */
    {"black",BLACK},            /* dark colors */
    {"blue",BLUE},
    {"green",GREEN},
    {"cyan",CYAN},
    {"red",RED},
    {"magenta",MAGENTA},
    {"brown",BROWN},
    {"lightgray",LIGHTGRAY},
    {"lightgrey",LIGHTGRAY},
    {"gray",LIGHTGRAY},
    {"grey",LIGHTGRAY},
    {"darkgray",DARKGRAY},            /* light colors */
    {"darkgrey",DARKGRAY},            /* light colors */
    {"lightblue",LIGHTBLUE},
    {"lightgreen",LIGHTGREEN},
    {"lightcyan",LIGHTCYAN},
    {"lightred",LIGHTRED},
    {"lightmagenta",LIGHTMAGENTA},
    {"yellow",YELLOW},
    {"white",WHITE}
    };

struct ckey internal_variables[] = {
  {"bounding_box_color",V_BOUNDING_BOX_COLOR},
  {"detorus_epsilon",V_DETORUS_EPSILON},
  {"high_constraint",V_HIGH_CONSTRAINT},
  {"high_boundary",V_HIGH_BOUNDARY},
  {"autochop_length",V_AUTOCHOP_LENGTH},
  {"string_curve_tolerance",V_STRING_CURVE_TOLERANCE},
  {"corona_state",V_CORONA_STATE},
  {"mpi_maxtask",V_MPI_MAXTASK}, 
  {"this_task",V_THIS_TASK}, /* MPI task number */
  {"window_aspect_ratio",V_WINDOW_ASPECT_RATIO},
  {"mindeg_debug_level",V_MINDEG_DEBUG_LEVEL},
  {"mindeg_min_region_size",V_MINDEG_MIN_REGION_SIZE},
  {"mindeg_margin",V_MINDEG_MARGIN},
  {"fix_count",V_FIX_COUNT},
  {"unfix_count",V_UNFIX_COUNT},
  {"edge_delete_count",V_EDGE_DELETE_COUNT},
  {"facet_delete_count",V_FACET_DELETE_COUNT},
  {"vertex_dissolve_count",V_VERTEX_DISSOLVE_COUNT},
  {"edge_dissolve_count",V_EDGE_DISSOLVE_COUNT},
  {"facet_dissolve_count",V_FACET_DISSOLVE_COUNT},
  {"body_dissolve_count",V_BODY_DISSOLVE_COUNT},
  {"edge_refine_count",V_EDGE_REFINE_COUNT},
  {"facet_refine_count",V_FACET_REFINE_COUNT},
  {"vertex_pop_count",V_VERTEX_POP_COUNT},
  {"edge_pop_count",V_EDGE_POP_COUNT},
  {"pop_tri_to_edge_count",V_POP_TRI_TO_EDGE_COUNT},
  {"pop_edge_to_tri_count",V_POP_EDGE_TO_TRI_COUNT},
  {"pop_quad_to_quad_count",V_POP_QUAD_TO_QUAD_COUNT},
  {"edgeswap_count",V_EDGESWAP_COUNT},
  {"t1_edgeswap_count",V_T1_EDGESWAP_COUNT},
  {"ps_labelsize",V_PS_LABELSIZE},
  {"ps_stringwidth",V_PS_STRINGWIDTH},
  {"ps_fixededgewidth",V_PS_FIXEDEDGEWIDTH},
  {"ps_tripleedgewidth",V_PS_TRIPLEEDGEWIDTH},
  {"ps_conedgewidth",V_PS_CONEDGEWIDTH},
  {"ps_bareedgewidth",V_PS_BAREEDGEWIDTH},
  {"ps_gridedgewidth",V_PS_GRIDEDGEWIDTH},
  {"scrollbuffersize",V_SCROLLBUFFERSIZE},
  {"visibility_debug",V_VISIBILITY_DEBUG},
  {"check_count",V_CHECK_COUNT},
  {"bad_next_prev_count",V_BAD_NEXT_PREV_COUNT},
  {"inconsistent_bodies_count",V_INCONSISTENT_BODIES_COUNT},
  {"edge_loop_count",V_EDGE_LOOP_COUNT},
  {"edges_same_vertices_count",V_EDGES_SAME_VERTICES_COUNT},
  {"facets_same_vertices_count",V_FACETS_SAME_VERTICES_COUNT},
  {"bad_error_count",V_BAD_ERRORS_COUNT}, 

  {"breakflag",V_BREAKFLAG},
  {"everything_quantities",V_EVERYTHING_QUANTITIES},
  {"gravity_constant",GRAV_CONST_TOK},
  {"hessian_slant_cutoff",V_HESSIAN_SLANT_CUTOFF},
  {"ambient_pressure_value",V_AMBIENT_PRESSURE},
  {"last_error",V_LAST_ERROR},
  {"memory_arena",V_MEMARENA},
  {"memory_used",V_MEMUSED},
  {"background",V_BACKGROUND},
  {"brightness",V_BRIGHTNESS},
  {"diffusion_coeff",V_DIFFUSION},
  {"transform_count",V_TRANSFORM_COUNT},
  {"scale_limit",V_SCALE_LIMIT},
  {"clock",V_CLOCK},
  {"cpu_counter",V_CPU_COUNTER},
  {"target_tolerance",V_TARGET_TOLERANCE},
  {"thickness",V_THICKNESS},
  {"spring_constant",V_GAP_CONSTANT},
  {"gap_constant",V_GAP_CONSTANT},
  {"lagrange_order",V_LAGRANGE_ORDER},
  {"last_eigenvalue",V_LAST_EIGENVALUE},
  {"last_hessian_scale",V_LAST_HESSIAN_SCALE},
  {"integral_order",V_INTEGRAL_ORDER},
  {"integral_order_1d",V_INTEGRAL_ORDER_1D},
  {"integral_order_2d",V_INTEGRAL_ORDER_2D},
  {"integration_order",V_INTEGRAL_ORDER},
  {"integration_order_1d",V_INTEGRAL_ORDER_1D},
  {"integration_order_2d",V_INTEGRAL_ORDER_2D},
  {"random_seed",V_RANDOM_SEED},
  {"random",V_RANDOM},
  {"linear_metric_mix",V_LINEAR_METRIC_MIX},
  {"quadratic_metric_mix",V_QUADRATIC_METRIC_MIX},
  {"pickvnum",V_PICKVNUM},
  {"pickenum",V_PICKENUM},
  {"pickfnum",V_PICKFNUM},
  {"eigen_pos",V_EIGENPOS},
  {"eigen_neg",V_EIGENNEG},
  {"eigen_zero",V_EIGENZERO},
  {"eigenpos",V_EIGENPOS},
  {"eigenneg",V_EIGENNEG},
  {"eigenzero",V_EIGENZERO},
  {"total_time",V_TIME},
  {"jiggle_temperature",V_JIG_TEMP},
  {"iteration_counter",V_ITER_COUNTER},
  {"scale_scale",V_SCALE_SCALE},
  {"vertex_count",V_VERTEXCOUNT},
  {"edge_count",V_EDGECOUNT},
  {"facet_count",V_FACETCOUNT},
  {"body_count",V_BODYCOUNT},
  {"facetedge_count",V_FACETEDGECOUNT},
  {"total_energy",V_ENERGY},
  {"total_area",V_AREA},
  {"total_length",V_LENGTH},
  {"scale",V_SCALE},
  {"space_dimension",V_SPACE_DIMENSION},
  {"surface_dimension",V_SURFACE_DIMENSION},
  {"torus",V_TORUS},
  {"symmetry_group",V_SYMMETRY_GROUP},
  {"simplex_representation",V_SIMPLEX},
  {"constraint_tolerance",V_TOLERANCE},
  {"hessian_epsilon",V_HESS_EPSILON},
  {"equi_count",V_EQUI_COUNT},
  {"delete_count",V_DELETE_COUNT},
  {"refine_count",V_REFINE_COUNT},
  {"notch_count",V_NOTCH_COUNT},
  {"dissolve_count",V_DISSOLVE_COUNT},
  {"pop_count",V_POP_COUNT},
  {"where_count",V_WHERE_COUNT}
 };
%}

   /* %option array */
D     [0-9]
PD    [1-9]
H     [0-9A-Fa-f]
E     [DEde][+-]?{D}+
W     [ \t\r\032]

%p 4000
%a 3000
%%
"/*"   { /* eat comment */
         register int c;
         in_comment = 1;
         for ( ; ; )
          { while ( (c = input()) != '*' && c != 0 )
          if ( c == '\n' ) line_no++; /* eat up text of comment */
            if ( c == '*' )
              { while ( (c = input()) == '*' ) ;
                if ( c == '/' ) break;    /* found the end */ 
            if ( c == '\n' ) line_no++; 
              }
            if ( c == 0 && yywrap() )
              { kb_error(2325,"End-of-file in comment\n",DATAFILE_ERROR ); break; }
          }
         in_comment = 0; 
       }

"//".*    /* comment */ ;

\n           { line_no++; }
_anti_line_no_TOK { errspot -= 15; line_no--; }

^{W}*-?{D}+  { char *c = strtok(yytext,whitespace);
               yylval.r = atof(strtok(yytext,whitespace));
               yylval.i = atoi(strtok(yytext,whitespace));
               if ( lists_flag==LISTS_FULL ) return(tok=LEAD_INTEGER_TOK); 
               else if ( uminus_flag || c[0] != '-' )  
                 return(tok = (fabs(yylval.r) > MAXINT) ? REAL_TOK: INTEGER_TOK);
               else { unput_string(c+1);  c[1] = 0;
               return (tok = minus_type(c[0])); 
             }
          }
{W}+[+-]{D}+ { char *c = strtok(yytext,whitespace);
               yylval.i = atoi(c); 
               yylval.r = atof(strtok(c,whitespace));
               yylval.qnum = 0;
               if ((lists_flag!=LISTS_OFF) && uminus_flag && (parens==0))
               return(tok = (fabs(yylval.r) > MAXINT) ? REAL_TOK: INTEGER_TOK);
               else { unput_string(c+1);  c[1] = 0;
               return (tok = minus_type(c[0])); 
             }
          }
{D}+      { yylval.i = atoi(yytext); 
            yylval.r = atof(strtok(yytext,whitespace));
            yylval.qnum = 0;
            return(tok = (fabs(yylval.r) > MAXINT) ? REAL_TOK: INTEGER_TOK);
          }
^{W}*{D}+"@"{D}+ { yylval.i = atoi(yytext);
              yylval.qnum = atoi(strchr(yytext,'@')+1);
              return(tok = LEAD_INTEGER_AT_TOK);
            }
{W}+-?{D}+"@"{D}+ { yylval.i = atoi(yytext);
              yylval.qnum = atoi(strchr(yytext,'@')+1);
              return(tok = INTEGER_AT_TOK);
            }

0x{H}+      { sscanf(yytext+2,"%x",&yylval.i); 
              yylval.r = (REAL)(yylval.i);
              return(tok = INTEGER_TOK);
            }  /* hex */
[01]+[Bb]   { char *c = yytext;  /* binary */
              yylval.i = 0;
              while ( isdigit(*c) ) { yylval.i = 2*yylval.i + *c - '0'; c++;}
              yylval.r = (REAL)(yylval.i);
              return(tok = INTEGER_TOK);
            }

{D}+"."{D}+[a-z]+   return VERSIONTOKEN_TOK;

^{W}*[+-]{D}+"."{D}*({E})?  |
^{W}*[+-]{D}*"."{D}+({E})?  |
^{W}*[+-]{D}+{E}            |
{W}+[+-]{D}+"."{D}*({E})?   |
{W}+[+-]{D}*"."{D}+({E})?   |
{W}+[+-]{D}+{E}   { 
             char *c = strtok(yytext,whitespace);
             yylval.r = atof(c); 
             if ((lists_flag!=LISTS_OFF) && uminus_flag && (parens==0))
               return(tok = REAL_TOK); 
             else 
             { unput_string(c+1);   c[1] = 0;
               verb_flag = 0;  
               return (tok = minus_type(c[0])); 
             }
          }

^{W}*{D}+"."{D}*({E})?   |
^{W}*{D}*"."{D}+({E})?   |
^{W}*{D}+{E}        |
{D}+"."{D}*({E})?   |
{D}*"."{D}+({E})?   |
{D}+{E}    { yylval.r = atof(yytext);
     return(tok = REAL_TOK); }

\"          { /* read quoted string */
              int n;
          in_quote = 1;
          for (n=0;;n++)
          { int c;
        c = input();
        if ( c == '\n' ) line_no++;
        if ( c == '"' && (n == 0 || (yytext[n-1] != '\\')) ) break; 
        if ( n < 198 ) yytext[n] = c;
        else if ( n==199 )
          kb_error(2326,"Quoted string over 198 characters.",WARNING);
              }
          yytext[n] = 0;
          reduce_string(yytext);
          in_quote = 0;
          return(tok = QUOTATION_TOK);
        }

{W}+-{W}+  verb_flag = 0; return (tok = minus_type('-'));   
{W}+-$     verb_flag = 0; return (tok = minus_type('-'));   

{W}+-      { if (datafile_flag && uminus_flag && (parens == 0) )  
               return(tok = UMINUS_TOK);
             else { verb_flag = 0; return ( tok = minus_type('-') );}
           }
{W}+-=     { verb_flag = 2; yylval.i = SUBASSIGN_OP; return (tok= ASSIGNOP_TOK);}

^-{W}      { return ( tok = minus_type('-')); }
^-         { if (datafile_flag && uminus_flag && (parens == 0) ) 
               return ( tok = UMINUS_TOK);
             else { verb_flag = 0; return ( tok = minus_type('-') ); }
           }
":="       { verb_flag = 2; return (tok= ASSIGN_TOK);}
";="       { verb_flag = 2; 
             kb_error(2327,"You mistyped ';=' for ':='?\n",WARNING);
             return (tok= ASSIGN_TOK);
           }
"+="        { verb_flag = 2; yylval.i = PLUSASSIGN_OP; return (tok= ASSIGNOP_TOK);}
"-="        { verb_flag = 2; yylval.i = SUBASSIGN_OP; return (tok= ASSIGNOP_TOK);}
"*="        { verb_flag = 2; yylval.i = MULTASSIGN_OP; return (tok= ASSIGNOP_TOK);}
"/="        { verb_flag = 2; yylval.i = DIVASSIGN_OP; return (tok= ASSIGNOP_TOK);}
"::="       { verb_flag = 2; return (tok = PERM_ASSIGN_TOK);}
":::="      { verb_flag = 2; return (tok = REDEFINE_TOK); }
-           { return (tok = minus_type('-'));  }
[<>+*/,.?%^\[\]]       { verb_flag = 0 ;  return(tok = yytext[0]); }
";"         { verb_flag = 1; return tok = ';'; }
"{"         {  verb_flag = 1; return(tok = yytext[0]); }
"}"         {  return(tok = yytext[0]);  }
"="         { return tok = (datafile_flag ? '=' : EQ_TOK); }
"("         {   verb_flag = 0; return(tok = yytext[0]); }
")"         {  return(tok = yytext[0]);  }
"`"         {  return(tok = yytext[0]);  }
"**"        { return(tok = '^'); }
"=="        { return(tok = EQ_TOK); }
"!="        { return(tok = NE_TOK); }
"<="        { return(tok = LE_TOK); }
">="        { return(tok = GE_TOK); }
"||"        { return(tok = OR_TOK); }
"&&"        { return(tok = AND_TOK); }
"!"         { return(tok = NOT_TOK); }
"|"         { return (tok = PIPE_TOK); /* pipe followed by quoted string */ }
">>"        { return (tok = REDIRECT_TOK); /* for redirection */ }
">>>"       { return (tok = REDIRECTOVER_TOK); /* for overwrite redirection */ }
"`"" "*","  { return (tok = BACKQUOTE_COMMA_TOK); }

#define    { if (!datafile_flag)
             kb_error(1880,"#define valid only in top of datafile.\n",WARNING);
             macro_flag = 1;
           }
#include{W}*\"[^"]*\"  { /* nested include */
             char *name = strchr(yytext,'"')+1;/* initial quote + 1 */
             *strchr(name,'"') = 0; /* replace final quote by 0 */
             push_commandfd(NULL,name);
           }

[A-Za-z_][A-Za-z0-9_]*   { strncpy(yylval.lexeme,yytext,LEXEME_SIZE);
                           if ( !macro() ) return identcase(yytext); 
                         } 

"'"[A-Za-z]"'"     { yylval.i = yytext[1]; return tok = QUOTED_LETTER_TOK; }
{W}      ;
":"     { if ( cond_expr_flag ) return tok = ':';}
.       { if ( isprint(yytext[0]) )
            sprintf(errmsg,"Illegal token: %c\n",yytext[0]);
          else
            sprintf(errmsg,"Illegal token: 0x%02X\n",yytext[0]);
          yyerror(errmsg);
          return tok=LEXERROR;
        }
%%

/* My own input() function to handle case insensitivity and macros */
/* and line counting */

/*********************************************************************
* function: savein()
* purpose:  for saving input for command definition 
*/
int inputbuffersize = 0;
int inputbufferspot = 0;  /* where next character will go */
int inputsave_flag = 0;  /* so only save when reading command */
char *inputbuffer;
void savein(int c)
{ if ( !inputsave_flag ) return;
  if ( inputbufferspot >= inputbuffersize )
  { int newsize = inputbuffersize?2*inputbuffersize:1000;
    inputbuffer = my_list_realloc(inputbuffer,newsize,ETERNAL_BLOCK);
    inputbuffersize = newsize;
  }
  inputbuffer[inputbufferspot++] = (char)c;
} // end savein()
 
/*********************************************************************
* function: reset_inputbuffer()
* purpose:  reset the input buffer to empty.
*/
void reset_inputbuffer()
{ inputbufferspot = 0;
  inputsave_flag = 1;
}

#define MOREMAX 1000
char morebuff[MOREMAX+2];

/*********************************************************************
* function: rawinput()
* purpose: get one character from input.
*/
int rawinput()
{
  int c=0,retval;

  if ( cmdptr ) /* read from string */
   {
     if ( *cmdptr == 0 )  /* newline at end of each line */
     { if ( previous_char != '\n' )
       { retval = previous_char = '\n';
         errbuff[errspot++] = retval; 
         savein(retval);
       }
       else retval = 0;
       goto rawreturn; 
     }
     c = *(cmdptr++);
     if ( c == MOREIN ) /* for really long input lines */
     { 
#ifdef USE_READLINE  //CSL
       prompt(CONTPROMPT,morebuff,MOREMAX);
#else
       getstring(morebuff,MOREMAX);
#endif
       cmdptr = morebuff;
       if ( !*cmdptr ) 
       { if (!in_quote && !in_comment) errbuff[errspot++] = '\n';  
         retval = '\n'; 
         savein(retval);
         goto rawexit; 
       }
       c = *(cmdptr++);
     }
   }
  else /* read from file */
   { for(;;)
     {
       if ( commandfd == NULL )  /* in case read after close */
       { retval = previous_char = 0; goto rawreturn; }
       c = getc(commandfd);
       if ( c != -1 ) break;
       pop_commandfd();  /* EOF, so pop #include stack */
       if ( !datafile_flag ) /* EOF for a command file */
       { retval =  previous_char = 0; goto rawreturn; }
     }
   }

  /* taking care of various line-ending combinations */
  if ( (c == '\n') && (previous_char == '\r') ) 
    return  rawinput();
  previous_char = c;
  if ( c == '\r' ) 
     c = '\n';

  errbuff[errspot++] = (char)c;
  retval = c;

  savein(retval);

rawexit:
  if ( errspot >= ERRBUFFSIZE ) /* need partial reset */
    { memcpy(errbuff,errbuff+ERRBUFFSIZE/2,ERRBUFFSIZE/2);
      errspot = ERRBUFFSIZE/2;
    }

rawreturn:
  return  retval;
} // end rawinput()

/***************************************************************************
* function: get_more_input()
* purpose: prompt user for more input.
*/

void get_more_input()
{int n; 
 #ifdef USE_READLINE //CSL
  prompt(MOREPROMPT,morebuff,MOREMAX);
 #else
  if ( !topflag && (!quiet_load_flag || (commandfd == stdin)) ) 
     outstring("more> ");
  getstring(morebuff,MOREMAX);
 #endif
  n = (int)strlen(morebuff); 
  if ( (morebuff[n-1] != MOREIN) && (morebuff[n-1] != '\n') )
  morebuff[n] = '\n'; morebuff[n+1] = 0;
  catfulltext(morebuff);
  cmdptr = morebuff;
}

/***************************************************************************
* function: kb_input()
* purpose: get next character from putback buffer or input.
*/
int kb_input()
{
  int c;

  if ( spot > 0 ) 
    { c = buff[--spot];
      goto input_exit;
    }

retry:
  c = rawinput();

  if ( c == '\\' )  /* line splicing */
  { /* have to do line splicing as input filter to remove 
       special character of line start */
    int cc = c;
    c = rawinput();
    if ( c == 0 ) 
    { get_more_input(); c = ' '; goto input_exit; }
    if ( c == '\n' ) 
    { int ccc;
      line_no++;
      ccc = rawinput();
      if ( ccc == 0 ) 
      { get_more_input(); c = ' '; goto input_exit; }
      rawunput(ccc); c = ' ';
    } 
    else { rawunput(c); c = cc; /* not linesplicing */ } 
  }
  if ( c == 0 && in_comment ) { get_more_input(); goto retry; }
  if ( c == 0 && in_quote ) { get_more_input(); goto retry; }
input_exit:

  return c;
} // end kb_input()

/*********************************************************************
* function: rawunput()
* purpose: put back one character to raw input stream.
*/
void rawunput(int c)
{
  if ( spot >= BUFFSIZE - 1 ) 
  { buff = my_list_realloc(buff,BUFFSIZE+500,ETERNAL_BLOCK);
    BUFFSIZE += 500;
  }

  buff[spot++] = (char)c;
} // end rawunput()

/*********************************************************************
* function: unput_string()
* purpose: put back string to input stream.
*/
void unput_string(char *s)
{ char *c;
  c = s ; while ( *c )  c++;   /* find end of token */
  while ( c != s ) { --c; unput(*c); }
} // end unput_string()

/*********************************************************************
* function: dump_buff()
* purpose: for error reporting 
*/
void dump_buff( 
  char *str,
  size_t size
)
{ int place;
  strcpy(str,"  Input line so far: \n");
  size -= (int)strlen(str)+3;
  errbuff[errspot] = 0;
  for ( place = errspot-1 ; place >= 0 ; place-- )
  { if (errbuff[place]==0 ) errbuff[place] = ' '; /* nulls creep in */ 
    if (errbuff[place]=='\n' && place < errspot-2) break;
  } 
  strncat(str,errbuff+place+1,size);
  if ( str[strlen(str)-1] != '\n' ) strcat(str,"\n");
} // end dump_buff()

/*********************************************************************
* function: yywrap()
* purpose: end of file handling for lex
*/
int yywrap()
{ /* Called at end of input.  Return 1 if really done, 0 if not */
  spot = 0; /* clean buffer */
  if ( (parens > 0) || (brace_depth > 0) || in_quote || in_comment 
         || in_function || in_control_structure /* || loopdepth */ )
    { get_more_input(); return 0; }
  if ( !help_flag )
    switch ( tok )
    { case '+': case '-': case '*': /* might be wrap symbols */
        if ( read_wrap_flag ) return 1;
        else { get_more_input(); return 0 ; }
        break;
      case UMINUS_TOK: case ',':
      case '=': case '?': case ':': case OR_TOK: case AND_TOK: case NOT_TOK:
      case EQ_TOK: case '>': case '<': case LE_TOK: case GE_TOK: case NE_TOK:
      case '/': case '%': case '^': case ON_CONSTRAINT_TOK: case VALUE_OF_CONSTRAINT_TOK:
      case HIT_CONSTRAINT_TOK: case ON_BOUNDARY_TOK: case DO_TOK: case FOR_TOK:
      case WHILE_TOK: case IF_TOK: case ELSE_TOK: case THEN_TOK: case SET_TOK:
      case DEFINE_TOK: case WHERE_TOK: case ASSIGN_TOK: case PERM_ASSIGN_TOK:
      case REDEFINE_TOK: case FUNCTION_TOK: case PROCEDURE_WORD_TOK:
      get_more_input(); return 0;
      break;
      default: return 1;
        }
  return 1;  /* done */
} // end yywrap()

/*************************************************************************
* function: macro_init()
* purpose: initialize macro definition list
*/
void macro_init()
{
  if ( macro_subs ) myfree((char *)macro_subs);
  if ( macros ) myfree((char *)macros);
  macro_subs = NULL;
  macros = NULL;
  macro_count = macro_subs_top = macro_max = macro_subs_max = 0;
} // end macro_init()

/*************************************************************************
* function: yylex_init()
* purpose: initialize lexical analyzer
*/
void yylex_init()
{
  unput_tok_count = 0;
  ubuff_spot = 0;
  spot = 0;
  errspot = 0;
  yylval.i = 0;
  yylval.r = 0.0;
  in_function = brace_depth = in_quote = in_comment = 0;
#ifdef FLEX_SCANNER
  if ( yy_current_buffer) YY_FLUSH_BUFFER;
#endif
} // end yylex_init()

/*************************************************************
*
*  Function: record_macro()
*
*  Purpose:  Record macro definition.
*/

void record_macro()
{
  int len;
  char *mspot;
  int cont_flag;
  int backslash_spot;

  macro_flag = 0;

  if ( macro_count >= macro_max )
  { if ( macro_max == 0 )
    { macro_max = 10; 
      macros = (struct macro *)mycalloc(macro_max,sizeof(struct macro));
    }
    else
    { 
      macros = (struct macro *)kb_realloc((char *)macros,
        2*macro_max*sizeof(struct macro));
      macro_max *= 2;
    }
  }

  strncpy(macros[macro_count].name,yytext,MACRONAMESIZE);

  if ( macro_subs_max == 0 )
  { macro_subs_max = 2*SUBMAX; 
    macro_subs = (char *)mycalloc(macro_subs_max,1);
  }

  /* read in macro string */
  for ( len = 0, cont_flag = 0 ; ; len++ )
    { int c = KB_INPUT();
      if ( len == 0 && (c == ' ' || c == '\t') )
      { len--;
        continue; /* skip leading whitespace */
      }
      if ( c == '\\' )
      { cont_flag = 1;
        backslash_spot = len;
      }
      else if ( c != ' ' && c != '\t' && c != '\n' && c != 0 )
        cont_flag = 0;
      if ( (c == '\0') || (c == '\n') )
      { 
        line_no++; 
        if ( !cont_flag ) 
           break; 
        cont_flag = 0;
        len = backslash_spot;
        macro_subs[macro_subs_top+len] = 0;
        len--;
        continue;
      }
      if ( macro_subs_top+len >= macro_subs_max-2 )
      { macro_subs = (char *)kb_realloc(macro_subs,macro_subs_max+2*SUBMAX);
        macro_subs_max += 2*SUBMAX;
      }
      macro_subs[macro_subs_top+len] = (char)c;
    }
  mspot = macro_subs+macro_subs_top;

  /* strip terminal comment */
  if ( strstr(mspot,"//") )
    { *strstr(mspot,"//") = 0;
      len = (int)strlen(mspot);
    }
  else mspot[len] = 0;
  /* strip trailing blanks */
  while ( (len > 0) && (mspot[len-1] == ' ') ) mspot[--len] = 0;
  macros[macro_count].subsize = len;
  macros[macro_count++].offset = macro_subs_top;
  macro_subs_top += len+1;
} // end record_macro()

/*************************************************************
*
*  Function:  macro()
*
*  Purpose: See if yytext is a macro, and do substitution.
*
*  Return:  1 if macro, 0 if not.
*/

int macro()
{
  int n,k;
  char *c;
  
  if ( macro_flag ) { record_macro(); return WAS_MACRO_DEF; }
  if ( strcmp(yytext,"_anti_line_no_") == 0 )
  { line_no--;  /* kludge line backup */
    errspot -= 15;
    return 1;
  }

  if ( !keep_macros_flag && !datafile_flag ) return 0;  /* no macros in commands */
  /* find name, simple linear search */
  for ( n = macro_count-1 ; n >= 0 ; n-- )  /* use latest definition */
     if ( strcmp(yytext,macros[n].name) == 0 ) break;

  if ( n < 0 ) return 0;   /* not found */

  /* insert string into input in reverse order so reads ok */
  c = macro_subs+macros[n].offset;
  for ( k = macros[n].subsize - 1 ; k >= 0 ; k-- )
    KB_UNPUT(c[k]);

  KB_UNPUT(' ');  /* substitution starts with whitespace; for proper
                     preservation of unary minus leading in the macro */

  return MACRO_EXPANDED;
} // end macro()

/*************************************************************************
* function: minus_type()
* purpose: for deciding type of minus sign or plus sign 
*/
int minus_type(int c /* '+' or '-' */)
{ int result;
  switch ( tok )
  {  case LEAD_INTEGER_TOK:
     case SIGNED_NUMBER_TOK:
     case INTEGER_TOK:
     case REAL_TOK:
     case PI_TOK:
     case E_TOK:
     case G_TOK:
     case PARAM_TOK: case ARRAYIDENT_TOK: case STRINGARRAY_TOK:
     case COORD_TOK: case VALUE_TOK: case BARE_TOK:
     case IDENT_TOK: case QUANTITY_NAME_TOK: case WRAP_TOK: case PERM_IDENT_TOK:
     case LENGTH_TOK: case DIHEDRAL_TOK: case VALENCE_TOK:
     case AREA_TOK: case VOLUME_TOK: case DENSITY_TOK: case TAG_TOK:
     case ID_TOK: case ORIGINAL_TOK: case SQ_MEAN_CURV_TOK:
     case OID_TOK: case INTERNAL_VARIABLE_TOK: case COLOR_TOK:
     case TOGGLEVALUE_TOK: case EXTRA_ATTRIBUTE_TOK: case PRESSURE_TOK:
     case ARRAY_ATTRIBUTE_TOK:
     case FRONTCOLOR_TOK: case BACKCOLOR_TOK: case TARGET_TOK:
     case NO_REFINE_TOK: case VOLCONST_TOK: case FRONTBODY_TOK: case NONCONTENT_TOK:
     case BACKBODY_TOK: case TETRA_POINT_TOK: case TRIPLE_POINT_TOK:
     case MIDV_TOK: case PHASE_TOK: case MODULUS_TOK: 
     case METHOD_NAME_TOK: case NODISPLAY_TOK: case HIT_PARTNER_TOK:
     case CENTEROFMASS_TOK:
     case ']': case ')':
        result = c; 
        break; 
     default: 
        result = (c=='-' ? UMINUS_TOK: UPLUS_TOK) ; 
        break;
  }
  return result; 
} // end minus_type()

/****************************************************************************
*
*  function identcase()
*
*  purpose: handle case of identifier token, maybe macro, keyword, etc.
*/
int keyword_compare(struct ckey *a, struct ckey *b)
{ return stricmp(a->name,b->name); }

int identcase(char *lexeme)
{
   int k,i;
   int type;
   struct sym *s;
   static int sorted = 0; /* flag for sorting keywords */
   struct ckey *keyptr;
   struct ckey key;  /* for bsearch key */
   char *c,*p;
   dll_func_type h;

   yylval.i = 0; 

   if ( ! sorted )
   { qsort((char*)datafile_keywords,
      sizeof(datafile_keywords)/ sizeof(struct dkey),sizeof(struct dkey),
      FCAST keyword_compare);
     qsort((char*)command_keywords,
      sizeof(command_keywords)/ sizeof(struct ckey),sizeof(struct ckey),
      FCAST keyword_compare);
     qsort((char*)togglenames,
      sizeof(togglenames)/ sizeof(struct ckey),sizeof(struct ckey),
      FCAST keyword_compare);
     qsort((char*)colornames,
      sizeof(colornames)/ sizeof(struct ckey),sizeof(struct ckey),
      FCAST keyword_compare);
     qsort((char*)internal_variables,
      sizeof(internal_variables)/ sizeof(struct ckey),sizeof(struct ckey),
      FCAST keyword_compare);
     qsort((char*)mathfunc_keywords,
      sizeof(mathfunc_keywords)/ sizeof(struct ckey),sizeof(struct ckey),
      FCAST keyword_compare);
     qsort((char*)mathfunc2_keywords,
      sizeof(mathfunc2_keywords)/ sizeof(struct ckey),sizeof(struct ckey),
      FCAST keyword_compare);
     sorted = 1;
   }

   if ( strcmp(lexeme,"_command_") == 0 )  { return tok = COMMAND_START_TOK; }
   if ( strcmp(lexeme,"_expr_") == 0 )   { return tok = EXPRESSION_START_TOK; }

   /* strip whitespace from front of lexeme */
   c = strtok(lexeme,whitespace);
   if ( c != lexeme ) { p = lexeme;  while ( *c )  *(p++) = *(c++); *p = 0;}

   if ( strlen(lexeme) == 1 )
   { char ch = lexeme[0];
     if ( verb_flag == 1 || // tok==PRINT_TOK||
       (verb_flag == 2 && (toupper(ch) < 'W'))  )
     { yylval.i=lexeme[0]; 
       if ( single_redefine[yylval.i].start ) return tok = SINGLE_REDEFD_TOK;
       switch ( yylval.i )
       { case 't': case 'l': case 'j': case 'P': case 'M':
         case 'w': case 'n': case 'm':  case 'b':
         case 'k': case 'K': case 'p': case 'y': case 'q':
           return ( tok = SINGLE_LETTER_ARG_TOK);
           case 'G': if ( verb_flag ) return ( tok = SINGLE_LETTER_ARG_TOK);
         else return tok=G_TOK;
         default: return ( tok = SINGLE_LETTER_TOK);
       } 
     }
     else 
      switch ( toupper(lexeme[0]) )
      { case 'X': case 'Y': case 'Z': case 'W':
          yylval.i = ((toupper(lexeme[0]) - 'X' + 4) % 4) + 1;
          return COORD_TOK;
        case 'P':
        { strncpy(idname,lexeme,sizeof(idname)); /* save text */
          yylval.qnum = V_PARAM_ATTR;
          yylval.etype = VERTEX;
          yylval.i = 1;
          return tok = ARRAY_ATTRIBUTE_TOK; 
        }
      }
    }

    // x1, x2, ... form of coordinates
    if ( toupper(lexeme[0]) == 'X' )
    { char *c;
      for ( c = lexeme+1 ; isdigit(*c) ; c++ ) ;
      if ( *c == 0 )
      { yylval.i = atoi(lexeme+1);
        if ( yylval.i != 0 )
          return COORD_TOK;
      }
    }

    // parameter names p1, p2, ...
    if ( toupper(lexeme[0]) == 'P' && isdigit(lexeme[1]) && !lexeme[2] )
    { yylval.i = atoi(lexeme+1);
      return PARAM_TOK;
    }

    // special gn kludge
    if ( lexeme[0] == 'g' )
    { char *c;
      for ( c = lexeme+1 ; isdigit(*c) ; c++ ) ;
      if ( *c == 0 )
      { yylval.i = atoi(lexeme+1);
        return GO_COUNT_TOK;
      }
    }

    if ( strncmp(lexeme,"usr",3) == 0 )
    { char *c;
      for ( c = lexeme+3 ; isdigit(*c) ; c++ ) ;
      if ( *c == 0 )
      { yylval.i = atoi(lexeme+3);
        return USERFUNC_TOK;
      }
    }
 
    key.name = lexeme;

#define BSEARCH(namelist)\
    (struct ckey*)bsearch((char*)&key,(char*)(namelist),\
         sizeof(namelist)/sizeof(struct ckey),\
          sizeof(struct ckey),FCAST keyword_compare);

    keyptr = BSEARCH(colornames);
    if ( keyptr )
        { yylval.r = yylval.i = keyptr->token;
          return (tok = INTEGER_TOK);
        }
    /* search math funcs */
    keyptr = BSEARCH(mathfunc_keywords);
    if ( keyptr ) {  yylval.i = keyptr->token; return (tok = MATHFUNC_TOK); }

    keyptr = BSEARCH(mathfunc2_keywords);
    if ( keyptr ) {  yylval.i  = keyptr->token; return (tok = MATHFUNC2_TOK); }

    if ( const_expr_flag )
    { for ( k=0 ; k<sizeof(const_expr_keywords)/ sizeof(struct ckey) ; k++ )
      if (stricmp(lexeme,const_expr_keywords[k].name)==0)
           return (tok = const_expr_keywords[k].token);
    }

    /* kludges for some backward compatibility */
    if ( strcmp(lexeme,"vertexnormal") == 0 )
      strcpy(lexeme,"vertex_normal");
    if ( lexeme[0] == '_' )
    {
      if ( strcmp(lexeme,"__vertex_normal") == 0 )
        strcpy(lexeme,"vertex_normal");
      if ( strcmp(lexeme,"__edge_vector") == 0 )
        strcpy(lexeme,"edge_vector");
      if ( strcmp(lexeme,"__facet_normal") == 0 )
        strcpy(lexeme,"facet_normal");
      if ( strcmp(lexeme,"__force") == 0 )
        strcpy(lexeme,"v_force");
      if ( strcmp(lexeme,"__velocity") == 0 )
        strcpy(lexeme,"v_velocity");
      if ( strcmp(lexeme,"__v_constraint_list") == 0 )
        strcpy(lexeme,"v_constraint_list");
      if ( strcmp(lexeme,"__e_constraint_list") == 0 )
        strcpy(lexeme,"e_constraint_list");
      if ( strcmp(lexeme,"__f_constraint_list") == 0 )
        strcpy(lexeme,"f_constraint_list");
     }
     
    if ( datafile_flag && !backquote_flag )
    { keyptr = BSEARCH(datafile_keywords);
      if ( keyptr ) {  return tok = keyptr->token;}
      keyptr = BSEARCH(command_keywords);
      if ( keyptr ) 
      { sprintf(errmsg,
         "Use of the keyword '%s' as an identifier is very ill-advised!\n",
           lexeme);
        kb_error(2328,errmsg, WARNING);
      }
    }
    else /* search keywords and internal variables */
    { 
      keyptr = BSEARCH(command_keywords);
      if ( keyptr ) 
      { if ( kb_stricmp(lexeme,"HELP") == 0 ) help_flag = 1;
        yylval.i = tok = keyptr->token;
        switch ( tok )
        { /* set verb_flag when expecting a command */
          case THEN_TOK: case ELSE_TOK: case DO_TOK: verb_flag = 1; break;
          /* clear verb_flag when expecting an expression */
          case PRINT_TOK: 
          case IF_TOK: case WHILE_TOK: case VIEW_TRANSFORMS_TOK: case TRANSFORM_DEPTH_TOK:
          case METIS_TOK: case KMETIS_TOK: case OMETIS_TOK: case EDGEWEED_TOK:
          case AREAWEED_TOK: case EDGEDIVIDE_TOK: case LANCZOS_TOK: case RITZ_TOK:
          case EIGENPROBE_TOK: case MOVE_TOK: case ZOOM_TOK: case LAGRANGE_TOK: case SET_TOK:
          case PRINTF_TOK: case LIST_TOK: case DELETE_TOK: case VERTEX_AVERAGE_TOK:
          case BINARY_PRINTF_TOK: 
          case DISSOLVE_TOK: case REFINE_TOK: case EDGESWAP_TOK: case FIX_TOK: case UNFIX_TOK:
              case SPRINTF_TOK: case EPRINT_TOK: case HISTOGRAM_TOK: case LOGHISTOGRAM_TOK:
          case FOREACH_TOK: case SHOW_EXPR_TOK: case UNSET_TOK:
          case HESSIAN_SADDLE_TOK: case HESSIAN_SEEK_TOK: case NOTCH_TOK:
          case AUTOCHOP_TOK: case AUTOPOP_TOK: case OPTIMIZE_TOK:
               verb_flag = 0; break;
          case SHOW_TOK: if ( verb_flag ) tok = SHOWVERB_TOK;
                      verb_flag = 0;
                      break;
          case TRANSFORM_EXPR_TOK: if ( verb_flag ) tok = TRANSFORM_EXPR_VERB_TOK;
                      verb_flag = 0;
                      break;
        }
         return tok;
      }
     }

    for ( i = 0 ; i < NUMDATATYPES ; i++ )
      if ( stricmp(lexeme,datatype_name[i]) == 0 )
      { yylval.i = DATATYPE_TOK;
        yylval.datatype = i;
        return tok = DATATYPE_TOK;
      }

    keyptr = BSEARCH(togglenames);
    if ( keyptr ) 
    { yylval.i = keyptr->token;
      if ( verb_flag != 0 ) tok = TOGGLENAME_TOK;
      else tok = TOGGLEVALUE_TOK;
      return tok;
    }
    keyptr = BSEARCH(internal_variables);
    if ( keyptr ) 
    { yylval.i = keyptr->token;
      return tok = INTERNAL_VARIABLE_TOK;
    }

    /* search local variables */
    yylval.i = lookup_local_var(lexeme);
    if ( yylval.i )
    { struct global *g;
      g = globals(yylval.i); 
      if ( g->flags & SUBROUTINE )
        return ( tok = PROCEDURE_TOK);
      else if ( g->flags & ORDINARY_PARAM )
      { switch ( g->type )
        { case VERTEX_TYPE: 
             yylval.etype = VERTEX;
             return ( tok = ELEMENT_IDENT_TOK);
          case ELEMENTID_TYPE:
             yylval.etype = -1;
             return ( tok = ELEMENT_IDENT_TOK);
          case EDGE_TYPE:
             yylval.etype = EDGE;
             return ( tok = ELEMENT_IDENT_TOK);
          case FACET_TYPE:
             yylval.etype = FACET;
             return ( tok = ELEMENT_IDENT_TOK);
          case BODY_TYPE:
             yylval.etype = BODY;
             return ( tok = ELEMENT_IDENT_TOK);
          case FACETEDGE_TYPE:
             yylval.etype = FACETEDGE;
             return ( tok = ELEMENT_IDENT_TOK);
          default:
            return ( tok = IDENT_TOK);
        }
      }
      else if ( g->flags & FUNCTION_NAME )
        return ( tok = FUNCTION_IDENT_TOK);
      else if ( g->flags & PROCEDURE_NAME )
        return ( tok = PROCEDURE_IDENT_TOK);
      else if ( g->flags & STRINGVAL )
        return ( tok = STRINGGLOBAL_TOK);
      else if ( g->flags & QUANTITY_NAME )  /* can't happen */
        return ( tok = QUANTITY_NAME_TOK);
      else if ( g->flags & METHOD_NAME )  /* can't happen */
        return ( tok = METHOD_NAME_TOK);
      else if ( g->flags & CONSTRAINT_NAME )  /* can't happen */
        return ( tok = CONSTRAINT_NAME_TOK);
      else if ( g->flags & BOUNDARY_NAME )  /* can't happen */
        return ( tok = BOUNDARY_NAME_TOK);
      else if ( g->flags & DYNAMIC_LOAD_FUNC )  /* can't happen */
        return ( tok = DYNAMIC_LOAD_FUNC_TOK);
      else if ( g->flags & ARRAY_PARAM )
	  {// if ( g->type == STRING )
	   //   return ( tok = STRINGARRAY_TOK);
		//else
          return ( tok = ARRAYIDENT_TOK);
      }
      else if ( g->flags & GLOB_LOCALVAR )
        return IDENT_TOK;
      else 
        return (tok = NEWIDENT_TOK);
    }

    /* search symbol table */
    s = symbol_lookup(lexeme);
    if ( s ) 
    { yysym = s; yylval.i = (int)(yysym-symtable); verb_flag = 0;
      return (tok = SYMBOL_TOK);
    }

    /* search parameter names */
    yylval.i = lookup_global_hash(lexeme,0,0,HASH_LOOK);
    if ( yylval.i != 0 )
    { int nametype = yylval.i & NAMETYPEMASK;
      struct global *g;
      yylval.i &= INDEXMASK; /* get plain index */

      if ( nametype == QUANTITYNAME )
        return tok = QUANTITY_NAME_TOK;
      else if ( nametype == METHODNAME )
        return tok = METHOD_NAME_TOK;
      else if ( nametype == PERM_NAME )
      { yylval.i |= PERMGLOBAL;
        if ( perm_globals(yylval.i)->flags & SUBROUTINE )
          return ( tok = PERM_PROCEDURE_TOK);
        else if ( perm_globals(yylval.i)->flags & STRINGVAL )
          return ( tok = PERM_STRINGGLOBAL_TOK);
        else if ( perm_globals(yylval.i)->flags & ARRAY_PARAM )
          return ( tok = ARRAYIDENT_TOK);
        else return (tok = PERM_IDENT_TOK);
      }
      yylval.i |= EPHGLOBAL;
      g = globals(yylval.i); 
      if ( g->flags & SUBROUTINE )
        return ( tok = PROCEDURE_TOK);
      else if ( g->flags & FUNCTION_NAME )
        return ( tok = FUNCTION_IDENT_TOK);
      else if ( g->flags & PROCEDURE_NAME )
        return ( tok = PROCEDURE_IDENT_TOK);
      else if ( g->flags & STRINGVAL )
        return ( tok = STRINGGLOBAL_TOK);
      else if ( g->flags & QUANTITY_NAME )
        return ( tok = QUANTITY_NAME_TOK);
      else if ( g->flags & METHOD_NAME )
        return ( tok = METHOD_NAME_TOK);
      else if ( g->flags & CONSTRAINT_NAME )
        return ( tok = CONSTRAINT_NAME_TOK);
      else if ( g->flags & BOUNDARY_NAME )
        return ( tok = BOUNDARY_NAME_TOK);
      else if ( g->flags & DYNAMIC_LOAD_FUNC )
        return ( tok = DYNAMIC_LOAD_FUNC_TOK);
      else if ( g->flags & ARRAY_PARAM )
        return ( tok = ARRAYIDENT_TOK);
      else 
      { switch ( g->type )
        { case VERTEX_TYPE: 
             yylval.etype = VERTEX;
             return ( tok = ELEMENT_IDENT_TOK);
          case ELEMENTID_TYPE:
             yylval.etype = -1;
             return ( tok = ELEMENT_IDENT_TOK);
          case EDGE_TYPE:
             yylval.etype = EDGE;
             return ( tok = ELEMENT_IDENT_TOK);
          case FACET_TYPE:
             yylval.etype = FACET;
             return ( tok = ELEMENT_IDENT_TOK);
          case BODY_TYPE:
             yylval.etype = BODY;
             return ( tok = ELEMENT_IDENT_TOK);
          case FACETEDGE_TYPE:
             yylval.etype = FACETEDGE;
             return ( tok = ELEMENT_IDENT_TOK);
          default:
            return ( tok = IDENT_TOK);
        }
      }
    }

    /* search extra attributes */
    for ( type = 0 ; type < NUMELEMENTS ; type++ )
    { struct extra *ex;
      int i;

      for ( i=0, ex=EXTRAS(type) ; i < web.skel[type].extra_count ; i++, ex++ )
        if ( stricmp(lexeme, ex->name) == 0 )
        { strncpy(idname,lexeme,sizeof(idname)); /* save text */
          yylval.qnum = i;
          yylval.etype = type;
          return tok = ex->array_spec.dim ? ARRAY_ATTRIBUTE_TOK: EXTRA_ATTRIBUTE_TOK; 
        }
    }

    /* search dynamic load libraries */
    h = search_libraries(lexeme);
    if ( h )
    { yylval.i = add_global(lexeme);
      globals(yylval.i)->flags |= DYNAMIC_LOAD_FUNC;
      globals(yylval.i)->value.funcptr = h;
      return ( tok = DYNAMIC_LOAD_FUNC_TOK);
    }

    /* if here, then not keyword */
    strncpy(idname,lexeme,sizeof(idname)); /* save text */
    if ( strlen(lexeme) == 1 )
    { yylval.i = lexeme[0];
      return ( tok = SINGLE_LETTER_TOK);
    }
    
   yylval.i = 0;
   return(tok = NEWIDENT_TOK) ;
} /* end identcase() */

/******************************************************************
*
*  function: keywordname()
*
*  purpose: find keyword of given token number. 
*
*/

char *keywordname(int toknum)
{  int k,i,imax;

    for ( k = 0 ; k < sizeof(command_keywords)/sizeof(struct ckey) ; k++ )
      if ( toknum == command_keywords[k].token )
      { static char name[82]; 
        strncpy(name,command_keywords[k].name,81);
        return name;
      }
    for ( k = 0 ; k < sizeof(togglenames)/sizeof(struct ckey) ; k++ )
      if ( toknum == togglenames[k].token )
      { static char name[82]; 
        strncpy(name,togglenames[k].name,81);
        return name;
      }
    for ( k = 0 ; k < sizeof(internal_variables)/sizeof(struct ckey) ; k++ )
      if ( toknum == internal_variables[k].token )
      { static char name[82]; 
        strncpy(name,internal_variables[k].name,81);
        return name;
      }

    imax = sizeof(colornames)/sizeof(struct ckey);
    for ( i = 0 ; i < imax ; i++ )
      if ( colornames[i].token == toknum )
        return colornames[i].name;

    imax = sizeof(datafile_keywords)/sizeof(struct dkey);
    for ( i = 0 ; i < imax ; i++ )
      if ( datafile_keywords[i].token == toknum )
        return datafile_keywords[i].name;

    imax = sizeof(mathfunc_keywords)/sizeof(struct ckey);
    for ( i = 0 ; i < imax ; i++ )
      if ( mathfunc_keywords[i].token == toknum )
        return mathfunc_keywords[i].name;

    imax = sizeof(mathfunc2_keywords)/sizeof(struct ckey);
    for ( i = 0 ; i < imax ; i++ )
      if ( mathfunc2_keywords[i].token == toknum )
        return mathfunc2_keywords[i].name;

   /* unfound */
   return tokname(toknum);

} // end keywordname()

/****************************************************************************
*
*  function identtype()
*
*  purpose: find type of an identifier. This version used by
*   is_variable() etc. commands.
*/

int identtype(char *word)
{
   int type;
   struct sym *s;
   struct ckey *keyptr;
   struct ckey key;  /* for bsearch key */


   if ( strlen(word) == 1 )
   {
     if ( single_redefine[yylval.i].start ) return tok = SINGLE_REDEFD_TOK;
     switch ( word[0] )
       { case 't': case 'l': case 'j': case 'P': case 'M':
         case 'w': case 'n': case 'm': case 'b':
         case 'k': case 'K': case 'p': case 'y':
           return ( tok = SINGLE_LETTER_ARG_TOK);
             case 'G': if ( verb_flag ) return tok = SINGLE_LETTER_ARG_TOK;
               else return tok = G_TOK; break;
         default: return ( tok = SINGLE_LETTER_TOK);
       } 
    }
    key.name = word;
    keyptr = BSEARCH(colornames);
    if ( keyptr )
          return  INTEGER_TOK;

    /* search math funcs */
    keyptr = BSEARCH(mathfunc_keywords);
    if ( keyptr ) {  return MATHFUNC_TOK; }

    keyptr = BSEARCH(mathfunc2_keywords);
    if ( keyptr ) {  return  MATHFUNC2_TOK; }

    /* search keywords and internal variables */
    { 
      keyptr = BSEARCH(command_keywords);
      if ( keyptr ) 
      {  
         return tok;
      }

      keyptr = BSEARCH(togglenames);
      if ( keyptr ) 
           return tok;
      keyptr = BSEARCH(internal_variables);
      if ( keyptr ) { return  INTERNAL_VARIABLE_TOK; }
    }

    /* search local variables */
    yylval.i = lookup_local_var(word);
    if ( yylval.i )
    { struct global *g;
      g = globals(yylval.i); 
      if ( g->flags & SUBROUTINE )
        return ( tok = PROCEDURE_TOK);
      else if ( g->flags & ORDINARY_PARAM )
        return ( tok = IDENT_TOK);
      else if ( g->flags & STRINGVAL )
        return ( tok = STRINGGLOBAL_TOK);
      else if ( g->flags & QUANTITY_NAME )  /* can't happen */
        return ( tok = QUANTITY_NAME_TOK);
      else if ( g->flags & METHOD_NAME )  /* can't happen */
        return ( tok = METHOD_NAME_TOK);
      else if ( g->flags & CONSTRAINT_NAME )  /* can't happen */
        return ( tok = CONSTRAINT_NAME_TOK);
      else if ( g->flags & BOUNDARY_NAME )  /* can't happen */
        return ( tok = BOUNDARY_NAME_TOK);
      else if ( g->flags & DYNAMIC_LOAD_FUNC )  /* can't happen */
        return ( tok = DYNAMIC_LOAD_FUNC_TOK);
      else if ( g->flags & ARRAY_PARAM )
        return ( tok = ARRAYIDENT_TOK);
      else if ( g->flags & GLOB_LOCALVAR )
        return IDENT_TOK;
      else return (tok = NEWIDENT_TOK);
    }

    /* search symbol table */
    s = symbol_lookup(word);
    if ( s ) 
    { yysym = s; yylval.i = (int)(yysym-symtable); verb_flag = 0;
      return (tok = SYMBOL_TOK);
    }

    /* search parameter names */
    yylval.i = lookup_perm_global(word);
    if ( yylval.i >= 0 )
      { 
        if ( perm_globals(yylval.i)->flags & SUBROUTINE )
        return ( tok = PERM_PROCEDURE_TOK);
        else if ( globals(yylval.i)->flags & STRINGVAL )
        return ( tok = PERM_STRINGGLOBAL_TOK);
        else return (tok = PERM_IDENT_TOK);
      }

    yylval.i = lookup_global(word);
    if ( yylval.i >= 0 )
      { 
        if ( globals(yylval.i)->flags & ORDINARY_PARAM )
          return ( tok = IDENT_TOK);
        if ( globals(yylval.i)->flags & SUBROUTINE )
          return ( tok = PROCEDURE_TOK);
        else if ( globals(yylval.i)->flags & STRINGVAL )
          return ( tok = STRINGGLOBAL_TOK);
        else if ( globals(yylval.i)->flags & QUANTITY_NAME )
          return ( tok = QUANTITY_NAME_TOK);
        else if ( globals(yylval.i)->flags & METHOD_NAME )
          return ( tok = METHOD_NAME_TOK);
        else if ( globals(yylval.i)->flags & DYNAMIC_LOAD_FUNC )
          return ( tok = DYNAMIC_LOAD_FUNC_TOK);
        else if ( globals(yylval.i)->flags & ARRAY_PARAM )
          return ( tok = ARRAYIDENT_TOK);
      }

    /* search extra attributes */
    for ( type = 0 ; type <= BODY ; type++ )
    { struct extra *ex;
      int i;

      for ( i = 0, ex = EXTRAS(type) ; 
         i < web.skel[type].extra_count ; i++ , ex++ )
      if ( stricmp(word, ex->name) == 0 )
      { strncpy(idname,word,sizeof(idname)); /* save text */
        yylval.qnum = i;
        yylval.etype = type;
        return tok = ex->array_spec.dim ? ARRAY_ATTRIBUTE_TOK: EXTRA_ATTRIBUTE_TOK; 
      }
    }

    /* if here, then not keyword */
    yylval.i = 0;
    return(tok = NEWIDENT_TOK) ;
} /* end identtype() */

/******************************************************************
 
   My own lexical analyzer.  Faster for my purposes and
   more comprehensible.
   
*******************************************************************/

/* States */
#define L_START  1
#define L_SLASH  2
#define L_LINECOMMENT 3
#define L_DOT 4
#define L_DIGITS 5
#define L_DIGITS_DOT 6
#define L_WHITESPACE 7
#define L_WHITESIGN 8
#define L_LINESTART 9
#define L_LEAD_INTEGER 10
#define L_LEAD_INTEGER_AT 11
#define L_INTEGER_AT 12
#define L_DIGITS_E 13
#define L_ALPHANUM 14
#define L_COLON 15
#define L_SEMICOLON 16
#define L_PLUS  17
#define L_MINUS 18
#define L_STAR  19
#define L_BANG  20
#define L_PIPE  21
#define L_AND   22
#define L_GREATER  23
#define L_COLONCOLON 24
#define L_SINGLE_QUOTE 25
#define L_HEXNUMBER   26
#define L_ZERO  27
#define L_GRGR  28
#define L_BACKQUOTE 29
#define L_SHARP  30
#define L_COLONCOLONCOLON 31
#define L_QUOTE 32
#define L_EQUAL 33
#define L_LESS  34
#define L_DIGITS_E_DIGITS 35
#define L_LEAD_MINUS 36
#define L_BACKSLASH 37
#define L_GRGRGR 38

/* Unput buffer */
int ubuff_max = 0;
int ubuff_spot = 0;  /* index of first vacant spot in unput_buff */
char *unput_buff;

/* maximum lexeme size */
int yytext_max;

/******************************************************************
*
* function: kb_input_new()
*
* purpose: fetch next input character from unput buffer or raw input.
*/
char kb_input_new()
{ char c;

  if ( ubuff_spot )
  { c = unput_buff[--ubuff_spot];
    unput_buff[ubuff_spot] = 0;
    return c;
  }
  c = rawinput();
  while ( c == 0 ) // done with this input
  { if ( yywrap() ) // check for continuation 
      return 0;   // no continuation, so done
    else
      c = rawinput();
   }
   return c;
   
} // end kb_input_new()

/******************************************************************
*
* function: kb_unput()
*
* purpose: put back lookahead character into input stream.
*/
void kb_unput( char c )
{ if ( ubuff_spot >= ubuff_max )
  { if ( unput_buff )
    { ubuff_max *= 2;
      unput_buff = realloc(unput_buff,ubuff_max);
    }
    else
    { ubuff_max = 16000;
      unput_buff = calloc(ubuff_max,sizeof(char));
    }
  }

  unput_buff[ubuff_spot++] = c;
  
} // end kb_unput()

/*********************************************************************
* 
* function: kblex()
*
* purpose: my own lexical analyzer, faster (for Evolver) and 
*          more comprehensible.
*
* return: next token number; lexeme is in yytext, and
*         other interesting data in yylval.
*/

int kblex()
{ int state;
  unsigned char nextchar;
  char *yyspot; /* first empty spot in yytext */
  int retval;

  PROF_START(kblex);

  if ( !yytext )
  { yytext_max = 200;
    yytext = calloc(yytext_max,sizeof(char));
  } 

    yyspot = yytext;
    memset(&yylval,0,sizeof(yylval));
    state = tok ? L_START : L_LINESTART;
    
    for(;;)
    { int len = yyspot-yytext;
      if ( len >= yytext_max-2 )
      { yytext_max *= 2;
        yytext = realloc(yytext,yytext_max);
        yyspot = yytext + len;
      }
      *(yyspot++) = nextchar = kb_input_new();
      switch ( state )
      {
        case L_START:
           if ( nextchar == '0' )
             state = L_ZERO;
           else if ( isdigit(nextchar) )
             state = L_DIGITS;
           else if ( isalpha(nextchar) ) 
             state = L_ALPHANUM;
           else
           switch ( nextchar )
           {
             case 0:  tok=0; goto kblex_exit;
             case '\n' : line_no++; yyspot = yytext; state = L_LINESTART;
                         break;
             case ' ': case '\t':
               yyspot--; state = L_WHITESPACE; break;
             case '/':  state = L_SLASH; break;
             case '.':  state = L_DOT; break;         
             case '_':  state = L_ALPHANUM; break;
             case ':':  state = L_COLON; break;
             case ';':  state = L_SEMICOLON; break;
             case '+':  state = L_PLUS; break;
             case '-':  state = L_MINUS; break; 
             case '*':  state = L_STAR; break; 
             case '!':  state = L_BANG; break; 
             case '|':  state = L_PIPE; break; 
             case '&':  state = L_AND; break; 
             case '>':  state = L_GREATER; break;
             case '\'': case 0x91: state = L_SINGLE_QUOTE; break;
             case '"': case 0x93: state = L_QUOTE; break;
             case '`':  state = L_BACKQUOTE; break;
             case '=':  state = L_EQUAL; break;
             case '<':  state = L_LESS; break;
             case '\\': state = L_BACKSLASH; break;
             case '#':  state = L_SHARP; break;

             case '(': verb_flag = 0;
                *yyspot = 0;  tok = nextchar; goto kblex_exit;
             case ')':
             case '^':
             case ',':
             case '?':
             case '%':
             case '[':
             case ']':
             case '}': 
             case '@':
                *yyspot = 0;  tok = nextchar; goto kblex_exit;
             case '{': verb_flag = 1;
                *yyspot = 0; tok = nextchar; goto kblex_exit;
             default:  
               sprintf(errmsg,"Illegal character '%c'\n",nextchar);
               kb_error(3658,errmsg,RECOVERABLE);
           }
           break;

        case L_BACKSLASH:
           /* allow trailing whitespace */
           while ( nextchar == ' ' || nextchar == '\t' )
              nextchar = kb_input_new();
           if ( nextchar == '\n' )
           { yyspot = yytext;
             state = L_START;
             line_no++;
             if ( datafile_flag != IN_DATAFILE )
               get_more_input();
             break;
           }
           kb_error(3659,
              "Illegal backslash. Backslash is line continuation character.\n",
                 RECOVERABLE);
           break;

        case L_EQUAL:
           if ( nextchar == '=' )
           { *yyspot = 0;
             tok = EQ_TOK;
             goto kblex_exit;
           }
           kb_unput(nextchar);
           yyspot[-1] = 0;
           tok = (datafile_flag ? '=' : EQ_TOK);
           goto kblex_exit;
           
        case L_QUOTE:
           { /* read quoted string */
             int n;
             in_quote = 1;
             for (n=0;;n++)
             { if ( nextchar == '\n' ) line_no++;
               if ( (nextchar == '"' || nextchar == 0x94) && (n == 0 || (yytext[n-1] != '\\')) ) 
                 break;
               if ( n >= yytext_max - 2 )
               { yytext_max *= 2;
                 yytext = realloc(yytext,yytext_max);
               }
               yytext[n] = nextchar;
               nextchar = kb_input_new();
             }
             yytext[n] = 0;
             reduce_string(yytext);
             in_quote = 0;
             tok = QUOTATION_TOK;
             goto kblex_exit;
           }
           break;
   
        case L_LESS:
           if ( nextchar == '=' )
           { *yyspot = 0;
             tok = LE_TOK;
             goto kblex_exit;
           }
           else
           { kb_unput(nextchar);
             yyspot[-1] = 0;
             tok = '<';
             goto kblex_exit;
           }
           break;

        case L_GREATER:
           if ( nextchar == '>' )
              state = L_GRGR;
           else if ( nextchar == '=' )
           { *yyspot = 0;
             tok = GE_TOK;
             goto kblex_exit;
           }
           else
           { kb_unput(nextchar);
             yyspot[-1] = 0;
             tok = '>';
             goto kblex_exit;
           }
           break;

        case L_GRGR:
           if ( nextchar == '>' )
           { *yyspot = 0;
             tok = REDIRECTOVER_TOK;
             state = L_GRGRGR;
           } 
		   else if ( nextchar == '2' )
           { *yyspot = 0;
             tok = REDIRECT_ERR_TOK;
             goto kblex_exit;
           } 
           else
           { kb_unput(nextchar);
             yyspot[-1] = 0;
             tok = REDIRECT_TOK;
             goto kblex_exit;
           }
           break;

        case L_GRGRGR:
           if ( nextchar == '2' )
           { *yyspot = 0;
             tok = REDIRECTOVER_ERR_TOK;
             goto kblex_exit;
           } 
           else
           { kb_unput(nextchar);
             yyspot[-1] = 0;
             tok = REDIRECTOVER_TOK;
             goto kblex_exit;
           }
           break;

        case L_SINGLE_QUOTE:
           if ( nextchar == '\'' || nextchar == 0x92 )
              kb_error(3577,"Empty single quotes.\n",RECOVERABLE);
           if ( nextchar == 0 || nextchar == '\n' )
              kb_error(3570,"Dangling single quote.\n",RECOVERABLE);
           yylval.i = nextchar;
           nextchar = kb_input_new();
           if ( nextchar != '\'' && nextchar != 0x92) 
              kb_error(3583,"Single quote not closed after one character.\n",
                              RECOVERABLE);
           yytext[1] = 0;
           tok = QUOTED_LETTER_TOK;
           goto kblex_exit;
           break;

        case L_BACKQUOTE:
           while ( nextchar == ' ' ) yyspot[-1] = nextchar = kb_input_new();
           if ( nextchar == ',' )
           { *yyspot = 0;
             tok = BACKQUOTE_COMMA_TOK;
             goto kblex_exit;
           }
           yyspot[-1] = 0;
           kb_unput(nextchar);
           tok = '`';
           goto kblex_exit;
           break;

        case L_SHARP:
           while ( isalpha(nextchar) )
              *(yyspot++) = nextchar = kb_input_new();
           yyspot[-1] = 0;
           if ( strcmp(yytext,"#define") == 0 )
           { if (!datafile_flag)
             kb_error(2853,"#define valid only in top of datafile.\n",WARNING);
             macro_flag = 1;
             state = L_START;
             yyspot = yytext;
             break;
           }
           if ( strcmp(yytext,"#include") == 0 )
           { 
             while ( (nextchar != '"' && nextchar != 0x94) && nextchar > 0 && nextchar != '\n' ) 
                nextchar = kb_input_new();
             if ( (nextchar != '"' && nextchar != 0x94) )
             { kb_error(6377,"Need double-quoted filename after #include.\n",UNRECOVERABLE);
             }
             yyspot = yytext;
             do
               *(yyspot++) = nextchar = kb_input_new();
             while ( (nextchar != '"' && nextchar != 0x94) && nextchar > 0 && nextchar != '\n');
             if ( (nextchar != '"' && nextchar != 0x94) )
             { kb_error(6378,"Unclosed quote after #include.\n",UNRECOVERABLE);
             }
             yyspot[-1] = 0;
             push_commandfd(NULL,yytext);
             state = L_LINESTART;
             yyspot = yytext;
             break;
           }         
		   sprintf(errmsg,"Illegal # directive, line %d.\n",line_no);
		   kb_error(3681,errmsg,RECOVERABLE);
           break;

        case L_ZERO:
           if ( toupper(nextchar) == 'X' )
             state = L_HEXNUMBER;
           else if ( nextchar == '.' )
             state = L_DIGITS_DOT;
           else if ( isdigit(nextchar) )
             state = L_DIGITS;
           else
           { kb_unput(nextchar);
             yyspot[-1] = 0;
             yylval.i = 0;
             verb_flag = 0;
             tok = INTEGER_TOK;
             goto kblex_exit;
           }
           break;

        case L_HEXNUMBER:
           if ( isdigit(nextchar) ||
                 (toupper(nextchar) >= 'A' && toupper(nextchar) <= 'F') )
              break;
           yyspot[-1] = 0;
           kb_unput(nextchar);
           sscanf(yytext+2,"%x",&yylval.i);
           yylval.r = (REAL)yylval.i;
           verb_flag = 0;
           tok = INTEGER_TOK;
           goto kblex_exit;

        case L_WHITESPACE:
           if ( nextchar == '+' || nextchar == '-' )
           { state = L_WHITESIGN;
             break;
           }
           if ( nextchar == ' ' || nextchar == '\t' )
           { yyspot--; break;
           }
           else
           { state = L_START;
             kb_unput(nextchar);
             yyspot--;
           }
           break;

        case L_WHITESIGN:
           kb_unput(nextchar);
           if ( !isspace(nextchar) && (yytext[0] == '-') )
           { if ( datafile_flag && uminus_flag && (parens == 0) )
             { tok = UMINUS_TOK;
               goto kblex_exit;
             }
           }
           /* else go back and treat as ordinary - */
           yyspot--;
           state = yytext[0] == '+' ? L_PLUS : L_MINUS;
           break;

        case L_LINESTART:
           if ( nextchar == 0 ) 
           { tok = 0;
             goto kblex_exit;
           }
           if ( nextchar == ' ' || nextchar == '\t' )
             yyspot--;
           else if ( isdigit(nextchar) )
             state = L_LEAD_INTEGER;
           else if ( nextchar == '-' )
             state = L_LEAD_MINUS;
           else
           { kb_unput(nextchar);
             state = L_START;
             yyspot = yytext;
           }
           break;
        
        case L_LEAD_MINUS:
           if ( isdigit(nextchar) )
             state = L_LEAD_INTEGER;
           else
           { kb_unput(nextchar);
             state = L_WHITESIGN;
             yyspot--;
           }
           break;

        case L_LEAD_INTEGER:
           if ( isdigit(nextchar) )
             break;
           if ( nextchar == '.' )
           { state = L_DIGITS_DOT;
             break;
           }           
           if ( toupper(nextchar) == 'X' )
           { state = L_HEXNUMBER;
             break;
           }
           if ( (toupper(nextchar) == 'E') || (toupper(nextchar) == 'D') )
           { state = L_DIGITS_E;
             break;
           }
           if ( nextchar == '@' )
           { state = L_LEAD_INTEGER_AT;
             break;
           }
           /* have LEAD_INTEGER_TOK*/
           kb_unput(nextchar);
           yyspot[-1] = 0;
           yylval.r = atof(yytext);
           yylval.i = atoi(yytext);
           verb_flag = 0;
           if ( lists_flag==LISTS_FULL ) 
           { tok = LEAD_INTEGER_TOK;
             goto kblex_exit;
           }
           else if ( uminus_flag || yytext[0] != '-' )
           { tok = (fabs(yylval.r) > MAXINT) ? REAL_TOK: INTEGER_TOK;
             goto kblex_exit;
           }
           else 
           { for ( yyspot -= 2 ; yyspot != yytext ; yyspot-- ) 
               kb_unput(*yyspot);
             yytext[1] = 0;
             tok = minus_type(yytext[0]);
             goto kblex_exit;
           }
           break;

        case L_DIGITS:
           if ( isdigit(nextchar) )
             break;
           if ( nextchar == '.' )
           { state = L_DIGITS_DOT;
             break;
           }
           if ( (toupper(nextchar) == 'E') || (toupper(nextchar) == 'D') )
           { state = L_DIGITS_E;
             break;
           }
           if ( nextchar == '@' )
           { state = L_INTEGER_AT;
             break;
           } 

           if ( toupper(nextchar) == 'B' )
           { char *c;
             yylval.i = 0;
             for ( c = yytext;  *c=='0' || *c=='1'  ; c++ )
               yylval.i = 2*yylval.i + (*c - '0');
             if ( c == yyspot-1 )
             { yylval.r = (REAL)yylval.i;
               verb_flag = 0;
               tok = INTEGER_TOK;
               goto kblex_exit;
             }
             /* else fall through to ordinary integer */
           }
           /* have INTEGER_TOK*/
           kb_unput(nextchar);
           yyspot[-1] = 0;
           yylval.r = atof(yytext);
           yylval.i = atoi(yytext);
           verb_flag = 0;
           tok = (fabs(yylval.r) > MAXINT) ? REAL_TOK: INTEGER_TOK;
           goto kblex_exit;

        case L_DIGITS_DOT:
           if ( (toupper(nextchar) == 'E') || (toupper(nextchar) == 'D') )
           { state = L_DIGITS_E;
             break;
           }
           if ( isdigit(nextchar) )
             break;
           if ( isalpha(nextchar) )
           { *yyspot = 0;
             tok = VERSIONTOKEN_TOK;
             goto kblex_exit;
           }
           /* have complete REAL */
           kb_unput(nextchar);
           yyspot[-1] = 0;
           yylval.r = atof(yytext);
           verb_flag = 0;
           tok = REAL_TOK;
           goto kblex_exit;

        case L_DOT:
           if ( isdigit(nextchar) )
           { state = L_DIGITS_DOT;
             break;
           }
           /* else just a dot */
           kb_unput(nextchar);
           yyspot[-1] = 0;
           tok = '.';
           goto kblex_exit;

        case L_LEAD_INTEGER_AT:
           if ( isdigit(nextchar) )
             break;
           /* have token */
           kb_unput(nextchar);
           yyspot[-1] = 0;
           yylval.i = atoi(yytext);
           yylval.qnum = atoi(strchr(yytext,'@')+1);
           if ( yylval.i == 0 )
           { sprintf(errmsg, "Missing task number, line %d.",line_no);
		     kb_error(3554,errmsg,DATAFILE_ERROR);
		   }
           verb_flag = 0;
           tok = LEAD_INTEGER_AT_TOK;
           goto kblex_exit;

        case L_INTEGER_AT:
           if ( isdigit(nextchar) )
             break;
           /* have token */
           kb_unput(nextchar);
           yyspot[-1] = 0;
           yylval.i = atoi(yytext);
           yylval.qnum = atoi(strchr(yytext,'@')+1);
           if ( yylval.i == 0 )
           {
		     sprintf(errmsg,"Missing task number. Line %d",line_no);
		     kb_error(3553,errmsg,DATAFILE_ERROR);
		   }
           verb_flag = 0;
           tok = INTEGER_AT_TOK;
           goto kblex_exit;

        case L_SLASH:
           switch ( nextchar )
           { case '*': 
              { /* eat comment */
                register int c;
                in_comment = 1;
                state = L_START;
                for ( ; ; )
                { while ( (c = kb_input_new()) != '*' && c != 0 )
                    if ( c == '\n' ) 
                    { line_no++; /* eat up text of comment */
                      state = L_LINESTART;
                    }
                  if ( c == '*' )
                  { while ( (c = kb_input_new()) == '*' ) ;
                    if ( c == '/' ) break;    /* found the end */
                    if ( c == '\n' ) 
                    { line_no++;
                      state = L_LINESTART;
                    }
                  }
                  if ( c == 0 && yywrap() )
                  { kb_error(2861,"End-of-file in comment\n",RECOVERABLE ); 
                    break; 
                  }
                }
                in_comment = 0;
                yyspot = yytext;
              }
              break;
            
            case '/': state = L_LINECOMMENT; break;
            case '=': *yyspot = 0; yylval.i = DIVASSIGN_OP; 
                       verb_flag = 0;
                       tok=ASSIGNOP_TOK;
                       goto kblex_exit;
                       
            default: kb_unput(nextchar); 
                     yyspot[-1] = 0;
                     verb_flag = 0;
                     tok = '/';
                     goto kblex_exit;
         }
         break;  /* end L_SLASH */

        case L_DIGITS_E:
          if ( isdigit(nextchar) || nextchar == '+' || nextchar == '-' )
          { state = L_DIGITS_E_DIGITS;
            break; 
          }       
		  sprintf(errmsg,"Missing exponent for scientific notation. Line %d\n",line_no);
          kb_error(3664, errmsg,  RECOVERABLE);
          break;

        case L_DIGITS_E_DIGITS:
          if ( isdigit(nextchar) ) 
            break;
          /* have token */
          kb_unput(nextchar);
          yyspot[-1] = 0;
          yylval.r = atof(yytext);
          verb_flag = 0;
          tok = REAL_TOK;
          goto kblex_exit;

        case L_ALPHANUM:
          if ( isalnum(nextchar) || nextchar == '_' )
            break;
          /* have token */
          kb_unput(nextchar);
          yyspot[-1] = 0;
          strncpy(yylval.lexeme,yytext,LEXEME_SIZE);
          retval = macro();
          if ( retval == WAS_MACRO_DEF ) 
          { /* macro() ate up rest of line */
            state = L_LINESTART;
            yyspot = yytext;
          }
          else if ( retval == MACRO_EXPANDED )
          { state = L_START;
            yyspot = yytext;
          }
          else
          {
            tok = identcase(yytext);
            goto kblex_exit;
          }
          break;

        case L_COLON:
          if ( nextchar == '=' )
          { verb_flag = 2; yyspot[-1] = 0; 
            tok = ASSIGN_TOK;
            goto kblex_exit;
          }
          if ( nextchar == ':' )
          { state = L_COLONCOLON;
            break;
          }
          kb_unput(nextchar);
          yyspot[-1] = 0;
          if ( cond_expr_flag )
          { tok = ':';
            goto kblex_exit;
          }
          /* else treat as whitespace */
          yyspot = yytext;
          state = L_START;
          break;

        case L_COLONCOLON:
          if ( nextchar == '=' )
          { verb_flag = 2; yyspot[-1] = 0; 
            tok = PERM_ASSIGN_TOK;
            goto kblex_exit;
          }
          if ( nextchar == ':' )
          { state = L_COLONCOLONCOLON;
            break;
          }          
		  sprintf(errmsg,"Illegal token '::'  Line %d\n",line_no);
		  kb_error(3568,errmsg,RECOVERABLE);
          break;

        case L_COLONCOLONCOLON:
          if ( nextchar == '=' )
          { verb_flag = 2; yyspot[-1] = 0; 
            tok = REDEFINE_TOK;
            goto kblex_exit;
          }          
		  sprintf(errmsg,"Illegal token ':::'  Line %d\n",line_no);
		  kb_error(3569,errmsg,RECOVERABLE);
          break;

        case L_SEMICOLON:
          if ( nextchar == '=' )
          { verb_flag = 2; 
            kb_error(2868,"You mistyped ';=' for ':='?\n",WARNING);
            tok= ASSIGN_TOK;
            goto kblex_exit;
          }
          kb_unput(nextchar);
          yyspot[-1] = 0;
          verb_flag = 1;
          tok = ';';
          goto kblex_exit;
 
        case L_PLUS:
          verb_flag = 0;
          if ( nextchar == '=' )
          { verb_flag = 0; *yyspot = 0;
            yylval.i = PLUSASSIGN_OP;
            tok = ASSIGNOP_TOK;
            goto kblex_exit;
          }
          if ( nextchar == '+' )
          { verb_flag = 0; *yyspot = 0;
            yylval.i = PLUSPLUS_OP;
            tok = INCREMENT_TOK;
            goto kblex_exit;
          }
          kb_unput(nextchar);
          yyspot[-1] = 0;
          tok = '+';
          goto kblex_exit;
          
        case L_MINUS:
          verb_flag = 0;
          if ( nextchar == '=' )
          { verb_flag = 0; yylval.i = SUBASSIGN_OP; 
            *yyspot = 0;
            tok= ASSIGNOP_TOK;
            goto kblex_exit;
          }
          if ( nextchar == '-' )
          { verb_flag = 0; *yyspot = 0;
            yylval.i = MINUSMINUS_OP;
            tok = INCREMENT_TOK;
            goto kblex_exit;
          }
          verb_flag = 0; 
          kb_unput(nextchar);
          tok = minus_type('-');
          goto kblex_exit;

        case L_STAR:
          verb_flag = 0;
          if ( nextchar == '=' )
          { verb_flag = 0; *yyspot = 0;
            yylval.i = MULTASSIGN_OP;
            tok = ASSIGNOP_TOK;
            goto kblex_exit;
          }
          if ( nextchar == '*' )
          { verb_flag = 0; *yyspot = 0; 
            tok = '^';
            goto kblex_exit;
          }
          kb_unput(nextchar);
          yyspot[-1] = 0;
          tok = '*';
          goto kblex_exit;

        case L_BANG:
           verb_flag = 0;
           if ( nextchar == '=' )
           { *yyspot = 0;
             tok = NE_TOK;
             goto kblex_exit;
           }
           kb_unput(nextchar);
           yyspot[-1] = 0;
           tok = NOT_TOK;
           goto kblex_exit;

        case L_PIPE:
           if ( nextchar == '|' )
           { *yyspot = 0;
             tok = OR_TOK;
             goto kblex_exit;
           }
           kb_unput(nextchar);
           yyspot[-1] = 0;
           tok = PIPE_TOK; 
           goto kblex_exit;

        case L_AND:
           if ( nextchar == '&' )
           { *yyspot = 0;
             tok = AND_TOK;
             goto kblex_exit;
           }         
		   sprintf(errmsg,"Illegal token '&'  Line %d\n",line_no);
		   kb_error(3566,errmsg,RECOVERABLE);
          break;

        case L_LINECOMMENT:
          if ( (nextchar == '\n') || (nextchar == 0) )
          { line_no++;
            state = L_LINESTART; 
            yyspot = yytext;
            break;
          }
          yyspot--;  /* don't save comment token */
          break;

        default: 
          fprintf(stderr,"Unhandled state: %d\n",state);
          *yyspot = 0;
          tok = 1;
          goto kblex_exit;
      }
    }
  kblex_exit:
  PROF_FINISH(kblex);
 
  return tok;
  
} // end kblex()
