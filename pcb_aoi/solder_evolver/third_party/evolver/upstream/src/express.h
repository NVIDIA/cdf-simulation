/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 * 
*************************************************************/

/********************************************************************
*
*  File: express.h
*
*  Contents: defines for expression parsing and evaluation.
*/

#ifdef __cplusplus
extern "C" {
#endif

/* node types, numbered above 1512 to avoid yacc token numbers */
#define GEN_ERROR_TOKEN 10000


/* for BREAK and CONTINUE */
extern int loopdepth;

/* tree node for expression trees */
struct treenode 
{
    int type;    /* type of node                     */
    int left;    /* left subexpression index offset  */
    int right;   /* right subexpression index offset */
    int line_no; /* line number of source file       */
    int file_no; /* number of source file            */
    int datatype; /* type of expression value        */
    int flags;
    union { int intval;            /* misc. integer data */
            int skipsize;          /* nodes to skip over */
            int argcount;          /* number of function arguments */
            int assigntype;        /* for assignments    */
            int aggrtype;          /* aggregate type     */
            int eltype;            /* element type       */
            int maxsteps;          /* for burchard()     */
            int coordnum;
            REAL  real;            /* constant value     */
            struct sym *symptr;    /* symbol table ptrs  */
            int localnum;          /* where element id stored*/
            int extranum;          /* number of extra attr */
            element_id id;         /* element id         */
            int name_id;           /* name identifier    */
            int letter;            /* for redefine       */
            int quant_id;          /* named quantity id  */
            int meth_id;           /* named method id    */
            int con_id;            /* number of constraint */
            int bdry_id;           /* number of boundary */
            int toggle_id;
            int toggle_state;      /* ON_ or OFF_        */ 
            int bool_val;          /* 0 or 1             */
            int intpow;            /* integer power      */
            int wherecount;
            int userfunc;
            char *string;          /* string value       */
            struct expnode enode;  /* for expression     */
            dll_func_type funcptr; /* DLL function       */
          } op1;    /* operand 1*/
    union { int intval;            /* misc. integer data */
            int eltype;            /* element type       */
            int valtype;           /* data type          */
            int breakdepth;        /* number of loop to break out of */
            struct sym *symptr;    /* symbol table ptrs  */
            char *string;
            int localnum;          /* where loop element id stored*/
            int coordnum;
            int attr_kind;
            int extranum;          /* number of extra attr */
            int jumpsize;
            int assigntype;
            int argcount;
            int name_id;
            int quant_id;
            int meth_id;
          } op2;    /* operand  2 */
    union { int intval[2]; /* misc. integer data */
            int breakjump;
            int argcount;
            int argtype;
            int extra_info;       /* extra attr eltype and number */
            int extranum;
            int connum;           /* constraint number */
            int bdrynum;
            int name_id;
            int localnum;          /* where element id stored*/
            int offset;
           } op3;
    union { int contjump;
            int skip;
            int arraydim;      // for ARRAYEXPR_NODE
            int argtype;
            int extranum;
            int ret_type;
            int eltype;            /* element type       */
            int name_id;
           } op4;
    union { struct sym *symptr;   /* symbol table ptrs  */
            char *string;
            struct locallist_t *locals; /* for procedures */
            int indexcount;   /* how many indices of array are fixed */
          } op5;
    int stack_delta;  /* what node does to runtime stack */
#ifdef _DEBUG
    int stack_spot;  /* where stack should be after evaluation
                                  of this node. */
#endif
    int stackpos; /* local variable on stack for storing stack
                     position for chopping stack after break or continue */
};
/* flags, also used for expnode */
#define LOCAL_VAR_REF 4
#define LOCAL_VAR_REF_2 8
#define LOCAL_VAR_REF_3 0x10
#define HAS_STRING    0x20
#define EPHEMERAL     0x40  /* refers to a nonpermanent name */
#define PERMNODE      0x80  /* part of permanent command */
#define HAS_LOCALLIST    0x100
#define HAS_STRING_5  0x200
#define DEALLOCATE_POINTER 0x400
#define IN_ELEMENT_LOOP  0x800
#define BREAKPOINT_NODE  0x1000
#define IS_RVALUE        0x2000
#define SET_ASSIGNOP     0x4000
#define RECALC_FLAG      0x8000
#define IS_VIRTUAL_ATTR  0x10000
#define DONT_RESIZE_FLAG 0x20000


/* for some bit packing */
#define ESHIFT 12

/* for some type and number packing */
#define YYTYPESHIFT 25
#define YYSHIFTMASK (((1<<YYTYPESHIFT)-1))

struct eval_frame {  /* for access to parent eval's */
   struct treenode **basenode;  /* for permload kludge */
   struct expnode base_ex;     /* for permload kludge and debugging */
   element_id self_id;          /* reference element, if any */
   int  parent_frame_spot;             /* top-most frame location */
   struct treenode *return_node; /* for procedure return address */
   int  flags; 
};

/* flag bits */
/* #define IN_ELEMENT_LOOP  0x0800  defined elsewhere in this file */
/*    set in frame flags only if frame is in elemernt loop on entry */

#define BASE_OF_EVAL     2
#define BASE_OF_WHOLE_STACK 4
/* Magic number to act as sentinel at end of stack */
#define STACKMAGIC 12738546.2341341

#ifdef __cplusplus
}
#endif

// Marker value used for array initializer syntax at runtime
#define ARRAYINIT_MARKER  1.451511416514134156e299

// Assignment operations
#define ASSIGN_OP     30001
#define PLUSASSIGN_OP 30002
#define SUBASSIGN_OP  30003
#define MULTASSIGN_OP 30004
#define DIVASSIGN_OP  30005
#define PLUSPLUS_OP   30006
#define MINUSMINUS_OP 30007

// Some parsing node types not picked up by node_extract.c
#define FUNCTION_PROTO_NODE 20001
#define PROCEDURE_PROTO_NODE 20002
#define ATTRIBUTE_NODE  20003
#define INIT_ELEMENT_NODE 20004
#define NEXT_ELEMENT_NODE 20005
#define INIT_SUBELEMENT_NODE 20006
#define INDEXED_ATTRIBUTE_NODE 20007
#define AGGREGATE_NODE 20008
#define EXPRESSION_START_NODE 20009


/* Some tokens used in top of datafile only */
#define CONTENT_RANK_TOK 35001
#define VOLUME_METHOD_NAME_TOK 35002
#define PARTNER_HITTING_TOK 35003
#define DISPLAY_ORIGIN_TOK 35004
#define LENGTH_METHOD_NAME_TOK 35005
#define AREA_METHOD_NAME_TOK 35006
#define HESSIAN_SPECIAL_NORMAL_VECTOR_TOK 35007
#define KEEP_ORIGINALS_TOK 35008
#define ELEMENT_MODULUS_TOK 35009
#define SWAP_COLORS_TOK 35010
#define ACTUAL_VOLUME_TOK 35011
#define KEEP_MACROS_TOK 35012
#define LAGRANGE_MULTIPLIER_TOK 35013
#define VERSION_TOK 35014
#define IGNORE_CONSTRAINTS_TOK 35015
#define IGNORE_FIXED_TOK 35016
#define LOAD_LIBRARY_TOK 35017
#define INTERP_BDRY_PARAM_TOK 35018
#define LAGRANGE_ORDER_TOK 35019
#define PARAMETER_1_TOK 35020
#define OPTIMIZING_PARAMETER_TOK 35021
#define V_EVERYTHING_QUANTITIES_TOK 35023
#define METHOD_TOK 35024
#define SCALAR_INTEGRAND_TOK 35025
#define VECTOR_INTEGRAND_TOK 35026
#define K_VEC_ORDER_TOK 35027
#define FORM_INTEGRAND_TOK 35028
#define MOBILITY_TOK 35029
#define MOBILITY_TENSOR_TOK 35030
#define BOUNDARY_CURVATURE_TOK 35031
#define GLOBAL_METHOD_TOK 35032
#define VIEW_TRANSFORM_GENS_TOK 35033
#define HOMOTHETY_TOK 35034
#define APPROX_CURV_TOK 35035
#define PHASEFILE_TOK 35037
#define AUTOPOP_TOK 35038
#define AUTOPOP_QUARTIC_TOK 35039
#define TOTAL_TIME_TOK 35040
#define EFFECTIVE_AREA_TOK 35041
#define RUNGE_KUTTA_TOK 35042
#define MEAN_CURV_INT_TOK 35043
#define NORMAL_CURVATURE_TOK 35044
#define SQUARE_CURVATURE_TOK 35045
#define SQGAUSS_TOK 35047
#define GAUSS_CURVATURE_TOK 35049
#define INSULATING_KNOT_ENERGY_TOK 35050
#define SPACE_DIMENSION_TOK 35051
#define SURFACE_DIMENSION_TOK 35052
#define SIMPLEX_REP_TOK 35053
#define CONFORMAL_TOK 35056
#define KLEIN_METRIC_TOK 35057
#define METRIC_TOK 35058
#define SYMMETRY_GROUP_TOK 35059
#define TORUS_TOK 35060
#define TORUS_FILLED_TOK 35061
#define SOAPFILM_TOK 35062
#define SYMMETRIC_CONTENT_TOK 35064
#define MEAN_CURV_TOK 35065
#define WULFF_TOK 35066
#define PERIODS_TOK 35067
#define DISPLAY_PERIODS_TOK 35068
#define CLIP_COEFF_TOK 35069    
#define SLICE_COEFF_TOK 35070
#define PARAMETERS_TOK 35071
#define SURFACE_ENERGY_TOK 35072
#define IMMEDIATE_AUTOPOP_TOK 35073
#define CONDUCTING_KNOT_ENERGY_TOK 35074
#define SCALE_LIMIT_TOK 35075
#define ZOOM_RADIUS_TOK 35076
#define ZOOM_VERTEX_TOK 35077
#define CONSTRAINT_TOLERANCE_TOK 35078
#define MERITFACTOR_TOK 35079
#define GRAV_CONST_TOK 35080
#define SPRING_CONSTANT_TOK 35081
#define TEMPERATURE_TOK 35082
#define UNPUTTED_TOK 35084
#define EFIXED_TOK 35085
#define FACES_TOK 35086
#define PARAMETER_FILE_TOK 35089
#define CONVEX_TOK 35090
#define NONWALL_TOK 35091
#define CONTENT_TOK 35094
#define VERSIONTOKEN_TOK 35095
#define VERTICES_PREDICTED_TOK 35096
#define EDGES_PREDICTED_TOK 35097
#define FACETS_PREDICTED_TOK 35098
#define BODIES_PREDICTED_TOK 35099
#define FACETEDGES_PREDICTED_TOK 35100
#define QUANTITIES_PREDICTED_TOK 35101
#define METHOD_INSTANCES_PREDICTED_TOK 35102
#define CALC_IN_3D_TOK 35103
#define PARAMETER_2_TOK 35104

// Miscellaneous
#define NO_TOKEN  0
#define ON_  15000
#define OFF_ 15001
#define METIS_MODE 15002
#define KMETIS_MODE 15003
#define LEXERROR    15004


// Internal variables

#define V_VERTEXCOUNT 1602
#define V_EDGECOUNT 1603
#define V_FACETCOUNT 1604
#define V_BODYCOUNT 1605
#define V_FACETEDGECOUNT 1606
#define V_ENERGY 1607
#define V_AREA 1608
#define V_LENGTH 1609
#define V_SCALE 1610
#define V_SURFACE_DIMENSION 1635
#define V_SPACE_DIMENSION 1636
#define V_TORUS         1638
#define V_TORUS_FILLED    1639
#define V_SYMMETRY_GROUP 1640
#define V_SIMPLEX      1641
#define V_INTEGRAL_ORDER 1642
#define V_TOLERANCE     1646
#define V_EQUI_COUNT    1648
#define V_DELETE_COUNT    1650
#define V_REFINE_COUNT    1651
#define V_NOTCH_COUNT    1652
#define V_DISSOLVE_COUNT    1653
#define V_POP_COUNT    1654
#define V_WHERE_COUNT    1655
#define V_HESS_EPSILON     1740
#define V_SCALE_SCALE 1778
#define V_ITER_COUNTER 1787
#define V_TIME 1795
#define V_JIG_TEMP 1796
#define V_EIGENPOS 1818
#define V_EIGENNEG 1819
#define V_EIGENZERO 1820
#define V_EVERYTHING_QUANTITIES_ 1822
#define V_PICKVNUM 1828
#define V_PICKENUM 1829
#define V_PICKFNUM 1830
#define V_LINEAR_METRIC_MIX 1832
#define V_RANDOM_SEED 1839
#define V_INTEGRAL_ORDER_1D 1841
#define V_INTEGRAL_ORDER_2D 1842
#define V_QUADRATIC_METRIC_MIX 1861
#define V_LAST_EIGENVALUE 1863
#define V_LAST_HESSIAN_SCALE 1864
#define V_LAGRANGE_ORDER 1869
#define V_GAP_CONSTANT 1877
#define V_THICKNESS 1878
#define V_TARGET_TOLERANCE 1879
#define V_CLOCK 1880
#define V_SCALE_LIMIT 1884
#define V_TRANSFORM_COUNT 1904
#define V_RANDOM 1929
#define V_BRIGHTNESS 1935
#define V_DIFFUSION 1936
#define V_BACKGROUND 1938
#define V_MEMARENA 1942
#define V_MEMUSED 1943
#define V_LAST_ERROR 1945
#define V_AMBIENT_PRESSURE 1975
#define V_HESSIAN_SLANT_CUTOFF 1976
#define V_CHECK_COUNT 1986
#define V_BREAKFLAG 1987
#define V_VISIBILITY_DEBUG  2018
#define V_SCROLLBUFFERSIZE  2019
#define V_PS_STRINGWIDTH 2025
#define V_PS_FIXEDEDGEWIDTH 2026
#define V_PS_TRIPLEEDGEWIDTH 2027
#define V_PS_CONEDGEWIDTH 2028
#define V_PS_BAREEDGEWIDTH 2029
#define V_PS_GRIDEDGEWIDTH 2030
#define V_DIFFUSION_  2094
#define V_VERTEX_DISSOLVE_COUNT 2095
#define V_EDGE_DISSOLVE_COUNT 2096
#define V_FACET_DISSOLVE_COUNT 2097
#define V_BODY_DISSOLVE_COUNT 2098
#define V_EDGE_REFINE_COUNT 2099
#define V_FACET_REFINE_COUNT 2100
#define V_VERTEX_POP_COUNT  2101
#define V_EDGE_POP_COUNT 2102
#define V_POP_TRI_TO_EDGE_COUNT 2104
#define V_POP_EDGE_TO_TRI_COUNT 2105
#define V_POP_QUAD_TO_QUAD_COUNT 2106
#define V_EDGESWAP_COUNT 2107
#define V_T1_EDGESWAP_COUNT 2108
#define V_SKINNY_REFINE_COUNT 2109
#define V_VERTEX_DELETE_COUNT 2110
#define V_EDGE_DELETE_COUNT 2111
#define V_FACET_DELETE_COUNT 2112
#define V_BODY_DELETE_COUNT 2113
#define V_FIX_COUNT 2114
#define V_UNFIX_COUNT 2115
#define V_PS_LABELSIZE 2116
#define V_CPU_COUNTER 2117
#define V_MINDEG_DEBUG_LEVEL 2268
#define V_MINDEG_MARGIN      2269
#define V_MINDEG_MIN_REGION_SIZE      2270
#define V_EVERYTHING_QUANTITIES 2271
#define V_BAD_NEXT_PREV_COUNT 2272
#define V_INCONSISTENT_BODIES_COUNT 2273
#define V_EDGE_LOOP_COUNT 2274
#define V_EDGES_SAME_VERTICES_COUNT 2275
#define V_FACETS_SAME_VERTICES_COUNT 2276
#define V_BAD_ERRORS_COUNT 2278
#define V_DETORUS_EPSILON 2308
#define V_HIGH_BOUNDARY 2307
#define V_HIGH_CONSTRAINT 2306
#define V_STRING_CURVE_TOLERANCE 2303
#define V_CORONA_STATE 2297
#define V_MPI_MAXTASK  2288
#define V_AUTOCHOP_LENGTH 2286
#define V_FACET_REVERSE_COUNT 2281
#define V_EDGE_REVERSE_COUNT 2280
#define V_THIS_TASK 2279
#define V_WINDOW_ASPECT_RATIO 2277
#define V_BOUNDING_BOX_COLOR 2309
