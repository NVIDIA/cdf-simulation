/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/

/***************************************************************** 
*
*  File: exprint.c
*
*  Purpose: To print user commands and functions in algebraic form
*          
*/

#include "include.h" 
#include "lex.h"
#include "ytab.h"
  
void check_room_left (size_t);
void print_quote (char*);
void linebreak (void);
void newline (void);

struct locallist_t *current_proc_locals[100];
int current_proc_depth;

/*************************************************
*
* function: assign_symbol()
*
* purpose: return string for assignment type
*/

char * assign_symbol (int sym)
{ switch ( sym )
  { case ASSIGN_OP: return " :="; 
    case PLUSASSIGN_OP: return " +=";
    case SUBASSIGN_OP: return " -="; 
    case MULTASSIGN_OP: return " *=";
    case DIVASSIGN_OP: return " /=";
  }
  return ""; 
} // end assign_symbol()

/*****************************************************************
*
*  Function print_express()
*
*  Purpose: print expression in algebraic format
*          Uses private working space, expanding as needed.
*          Returns pointer to working space.
*
*/

static char *pos;  /* current position in string */
static char vch;
static size_t  strsize;
static char *strstart;
static char *linestart;
static int  bracket_depth; 

/* precedence levels for knowing how to parenthesize */
#define PREC_INDEX    55
#define PREC_POW      50
#define PREC_UMINUS   45
#define PREC_MUL      40
#define PREC_DIV      40
#define PREC_MOD      37
#define PREC_ADD      35
#define PREC_SUB      35
#define PREC_COMP     30
#define PREC_NOT      25
#define PREC_AND      20
#define PREC_OR       15
#define PREC_COND     13
#define PREC_ASSIGN   10
#define PREC_ARG       0

/**************************************************************************
*
* Function: print_express()
*
* Purpose: Convert parse tree for expression to ASCII string in 
*          character array pos.
*/

char *print_express(
  struct expnode *node,    /* expression tree */
  int c     /* character for parameters */
)
{ char *src,*dest;
  int linestart_flag;

  if ( !strstart )  /* for first time thru */
    { strsize = 200;
      strstart = my_list_calloc(1,strsize,ETERNAL_BLOCK);
    }
  else strstart[0] = '\0';
  linestart = pos = strstart;
  vch = (char)c;

  if ( !node || !node->root ) { strcpy(strstart,"{}"); return strstart;}

  bracket_depth = 0;
  current_proc_locals[0] = node->locals;
  current_proc_depth = 0;
  exprint_recur(node->root,0);

  // Combine multiple spaces, except at start of line
  // Also eliminate spaces before commas
  linestart_flag = 1;
  for ( src = strstart, dest = strstart ; *src != 0 ; src++ )
  { if ( *src == '\n' ) linestart_flag = 1;
    else if ( linestart_flag && (*src != ' ') )
      linestart_flag = 0;
    if ( *src == ' ' && src[-1] == ' ' && !linestart_flag ) 
      continue;
    if ( *src == ',' && dest[-1] == ' ' )
    { dest[-1] = *src;
      continue;
    }
    *dest = *src;
     dest++; 
  }
  *dest = 0;
       
    
  return strstart;

} // end print_express()

/**********************************************************************
*
* Function: linebreak()
*
* Purpose: Insert a linebreak when the line gets too long.
*/

#define INDENT 2
#define GOODLEN 75
#define MINLEN  30
void linebreak()
{ int i;
  char *c,*cc;
  char *minline = linestart + MINLEN;
  int extra_indent = 0;

  if ( pos - linestart < GOODLEN ) return;

  cc = NULL;
 
  /* search for end of quote, starting at end so don't break quote */
  for ( c = pos-1 ; c != linestart ; c-- )
      if ( *c == '"' ) 
      { cc = (c[1] == ';' || c[1] == ',') ? c+1 : c; 
        if ( c[1] == ',' ) extra_indent = 2;
        break; 
      } 

  if ( cc == NULL ) /* search for end bracket */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
    if ( *c == '}' ) { cc = (c[1] == ';') ? c+1 : c; break; }
  if ( cc == NULL ) /* scan back for handy ';' */ 
    for ( c = linestart + GOODLEN ; c != minline ; c-- )
     if ( *c == ';' ) { cc = c;  break; }
  if ( cc == NULL ) /* just look for { */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( *c == '{' ) { cc = c; break; }
  if ( cc == NULL ) extra_indent = 2;
  if ( cc == NULL ) /* just look for space */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( *c == ' ' ) { cc = c; break; }
  if ( cc == NULL ) /* just look for comma */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( *c == ',' ) { cc = c; break; }
  if ( cc == NULL ) /* just look for = */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( *c == '=' ) { cc = c; break; }
  if ( cc == NULL ) /* just look for logic signs */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( (*c == '&') || (*c == '%') ) 
          { cc = c; break; }
  if ( cc == NULL ) /* just look for arithmetic signs */
   for ( c = linestart + GOODLEN ; c != minline ; c-- )
    if ( ((*c == '+') || (*c == '-')) &&  !(isalpha(c[-1]) && !isalpha(c[-2]))) 
          { cc = c; break; }              /* don't split scientific notation */
  if ( (cc == NULL) || (cc - linestart < GOODLEN/2) )
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( ((*c == '*') || (*c == '/')) && (c[1] != '=') && (c[1] != '*') ) 
         { cc = c; break; }
  if ( cc == NULL ) /* just look for closing parenthesis */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( *c == ')' ) { cc = c; break; }
  if ( cc == NULL ) /* just look for opening parenthesis */
     for ( c = linestart + GOODLEN ; c != minline ; c-- )
       if ( *c == '(' ) { cc = c; break; }
 
  /* Have break, so do split, with *cc as last character on old line */
  if ( cc ) 
  { char *ch;
    int bd = bracket_depth;
    linestart = cc+3; 
     c = cc+1 ; while (*c == ' ' ) c++ ;  /* skip spaces */
     for ( ch = c ; ch < pos ; ch++ ) /* unwind bracket depth */
       if ( *ch == '{' ) bd--;
       else if ( *ch == '}' ) bd++;
     pos = cc + 3 + INDENT*bd + extra_indent + strlen(c);
     kb_memmove(cc+3+INDENT*bd+extra_indent,c,strlen(c));
     cc[1] = (char)(bd ? ' ' : '\\');
     cc[2] = '\n';
     for ( i = 0 ; i < INDENT*bd + extra_indent ; i++ ) cc[i+3] = ' ';
     *pos = 0;
  }
  else
  { if ( bracket_depth <= 0 ) *(pos++) = '\\';
    *(pos++) = '\n';
    linestart = pos;
    for ( i = 0 ; i < INDENT*bracket_depth ; i++ ) *(pos++) = ' ';
    *pos = 0;
  }
} // end linebreak()

/***************************************************************************
*
* function:  newline()
*
* purpose:  Begin a new line, suitably indented.
*/
void newline()
{
  int i;
  *(pos++) = '\n';
  for ( i = 0 ; i < INDENT*bracket_depth ; i++ )
    *(pos++) = ' ';
  *pos = 0;
  linestart = pos;

} // end newline()

/**********************************************************************
*
* Function: check_room_left()
* 
* Purpose: Keep print string from overflowing.
*/

void check_room_left(size_t n  /* room needed */)
{
   /* check room remaining */
   if ( (pos + n - strstart) > (int)strsize )
    { size_t len = pos - strstart;
      size_t linespot = linestart - strstart;
      strstart = my_list_realloc(strstart,strsize + 1000 + n,ETERNAL_BLOCK); 
      strsize += 1000 + n;
      linestart = strstart + linespot;
      pos = strstart + len;
    }
} // end check_room_left()

/**********************************************************************
*
* Function: print_quote()
* 
* Purpose: Add quoted string to output string.
*/

void print_quote(char *c)
{ check_room_left(strlen(c)+30);
  convert_string(c,pos,msgmax);
  pos += strlen(pos);

} // end print_quote()

/********************************************************************
*
* function: convert_string()
*
* purpose: convert string from internal to printable representation.
*         Converts to C escapes, encloses in quotes.
*/

void convert_string(
  char *c,   /* source */
  char *p,    /* destination */
  int space /* in destination */
)
{ int count;
  /* convert to C escape sequences */
  *(p++) = '"';
  if ( c )
  { for ( count=5 ; *c && count<space; c++ )
    { switch ( *c )
      { case '\n': *(p++) = '\\'; *(p++) = 'n'; count += 2; break;
        case '\r': *(p++) = '\\'; *(p++) = 'r'; count += 2; break;
        case '\t': *(p++) = '\\'; *(p++) = 't'; count += 2; break;
        case '\b': *(p++) = '\\'; *(p++) = 'b'; count += 2; break;
        case '"': *(p++) = '\\'; *(p++) = '"'; count += 2; break;
        case '\\': *(p++) = '\\'; *(p++) = '\\'; count += 2; break;
        default: *(p++) = *c; count++;
      }
   }
   if ( *c )
     kb_error(4556,"String too long.\n",RECOVERABLE);
  }
  *(p++) = '"';
  *p = 0;
  return;
}   // end convert_string()

/*************************************************************************
*
* function: exprint_recur()
*
* purpose: Convert node of parse tree to ASCII, recursively doing sons.
*/

void exprint_recur( 
  struct treenode *node,
  int prec_parent  /* precedence level of parent, for parenthesizing */
)
{
  struct treenode *nn;
  struct extra *ex;
  struct locallist_t *localbase = current_proc_locals[current_proc_depth];

   check_room_left(1000); 
  /* insert some handy line breaks */
  if ( (pos - linestart > GOODLEN) /* && (pos[-2] != ';') */ ) 
    linebreak(); 
  switch ( node->type )
    { 
      case NOP_NODE:  return;
      case NULLBLOCK_NODE: sprintf(pos,"{}");pos+=2; return;
      case NULLCMD_NODE:  return;

      case BREAKPOINT_NODE:
         sprintf(pos,"breakpoint %s ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;
         
      case UNSET_BREAKPOINT_NODE:
         if ( node->left )
         { sprintf(pos,"unset breakpoint %s ",globals(node->op1.name_id)->name);
           pos += strlen(pos);
           exprint_recur(node+node->left,prec_parent);
         }
         else
         { sprintf(pos,"unset breakpoints");
           pos += strlen(pos);
         }
         return;
         
      case SUPPRESS_WARNING_NODE:
         sprintf(pos,"suppress_warning "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case UNSUPPRESS_WARNING_NODE:
         sprintf(pos,"unsuppress_warning "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case KEYLOGFILE_NODE:
         sprintf(pos,"keylogfile "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case LOGFILE_NODE:
         sprintf(pos,"logfile "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case CMDLIST_NODE:
         exprint_recur(node+node->left,prec_parent);
         if ( node->right )
         {  if ( pos[-1] == ' ' ) 
              pos--;
            sprintf(pos,"; "); pos += 2;
            newline();
            exprint_recur(node+node->right,prec_parent);
          }
         return;
     
      case BACKQUOTE_START_NODE: return;
      case BACKQUOTE_END_NODE:
         exprint_recur(node+node->right,prec_parent); /* left was dummy */
         return;

      case TEXT_SPOT_NODE:
         exprint_recur(node+node->left,prec_parent); 
         *(pos++) = ',';  *pos = 0;
         exprint_recur(node+node->right,prec_parent);
         return;

      case TEXT_SIZE_NODE:
         exprint_recur(node+node->left,prec_parent); 
         *(pos++) = ',';  *pos = 0;
         exprint_recur(node+node->right,prec_parent);
         return;

      case DISPLAY_TEXT_NODE:
         sprintf(pos,"display_text("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         *(pos++) = ',';  *pos = 0;
         exprint_recur(node+node->right,prec_parent);
         *(pos++) = ')';  *pos = 0;
         return;

      case DELETE_TEXT_NODE:
         sprintf(pos,"delete_text("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         *(pos++) = ')';  *pos = 0;
         return;

      case CONSTRAINT_FIXED_NODE:
         strcat(pos,"is_constraint["); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         strcat(pos,"].fixed "); pos += strlen(pos);
         break;

         case CONSTRAINT_NONNEGATIVE_NODE:
         strcat(pos,"is_constraint["); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         strcat(pos,"].nonnegative "); pos += strlen(pos);
         break;

     case CONSTRAINT_NONPOSITIVE_NODE:
         strcat(pos,"is_constraint["); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         strcat(pos,"].nonpositive "); pos += strlen(pos);
         break;

      case ACOMMANDEXPR_NODE:
         *(pos++) = '`';   *pos = 0;/* surround with backquotes */
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,";`");
         pos+=2; /* surround with backquotes; make sure of ; */
         if ( node->right )
          { sprintf(pos,", "); pos += 2;
            exprint_recur(node+node->right,prec_parent);
          }
         return;

      case ATTR_FUNCTION_END_NODE:
         exprint_recur(node+node->left,prec_parent);  /* define part */
         strcat(pos," function {"); pos += strlen(pos);
         exprint_recur(node+node->right,prec_parent);  /* code part */
         strcat(pos," } "); pos += strlen(pos);
         return;
     
      case ATTR_FUNCTION_NODE:
         exprint_recur(node+node->left,prec_parent);  /* define part */
         return;

      case WRAP_VERTEX_NODE:
         strcat(pos,"wrap_vertex("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case CREATE_VERTEX_NODE:
         sprintf(pos,"new_vertex("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case FACET_CROSSCUT_NODE:
         sprintf(pos,"facet_crosscut("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case CREATE_EDGE_NODE:
         sprintf(pos,"new_edge("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case CREATE_FACET_NODE:
         sprintf(pos,"new_facet("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case CREATE_BODY_NODE:
         sprintf(pos,"new_body"); pos += strlen(pos);
         return;

      case ELINDEX_NODE:
         exprint_recur(node+node->left,prec_parent);
         if ( node->right )
         { *pos = '@'; pos++; *pos = 0;
            exprint_recur(node+node->right,prec_parent);
         }
         return;

      case PUSH_ELEMENT_ID_NODE:
		  sprintf(pos,"%s%d@%d\n",(inverted(node->op1.id)?" -":""),
             node->op1.id & OFFSETMASK, id_task(node->op1.id));
         pos += strlen(pos);
         return;
         
      case VALID_ELEMENT_NODE:
         sprintf(pos,"valid_element(%s[",typenames[node->op1.eltype]); 
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,"])"); pos+=2;
         return;
   
      case VALID_CONSTRAINT_NODE:
         sprintf(pos,"valid_constraint("); 
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,")"); pos+=1;
         return;
   
      case VALID_BOUNDARY_NODE:
         sprintf(pos,"valid_boundary("); 
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,")"); pos+=1;
         return;
   
      case MATRIX_INVERSE_NODE:
         sprintf(pos,"matrix_inverse(%s,%s)",
            globals(node->op1.name_id)->name,globals(node->op2.name_id)->name);
         pos += strlen(pos);
         return;

      case MATRIX_MULTIPLY_NODE:
         sprintf(pos,"matrix_multiply(%s,%s,%s)",
            globals(node->op1.name_id)->name,globals(node->op2.name_id)->name,
            globals(node->op3.name_id)->name);
         pos += strlen(pos);
         return;

      case MATRIX_DETERMINANT_NODE:
         sprintf(pos,"matrix_determinant(%s)",
            globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;

      case MERGE_VERTEX_NODE:
         sprintf(pos,"vertex_merge("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case MERGE_EDGE_NODE:
         sprintf(pos,"edge_merge("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case MERGE_FACET_NODE:
         sprintf(pos,"facet_merge("); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,prec_parent);
         sprintf(pos,")"); pos++;
         return;

      case COMMAND_BLOCK_NODE:
         sprintf(pos,"{ "); pos+=2; bracket_depth++;
         exprint_recur(node+node->left,prec_parent);
         bracket_depth--;
         newline();
         sprintf(pos,"}"); pos++;
         return;

      case LOCAL_LIST_START_NODE:
         sprintf (pos,"local ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;

      case DECLARE_LOCAL_NODE:
       { 
         if ( node->left )
         { 
           exprint_recur(node+node->left,0);
           *pos = ','; pos++; *pos = 0;
         }
         sprintf (pos,"%s",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;
       }
      
      case DEFINE_QUANTITY_NODE:
       { struct gen_quant *g = GEN_QUANT(node->op1.quant_id);
         sprintf(pos,"/* Definition of quantity %s was originally here.*/",
            g->name);
         pos += strlen(pos);
         return;
       }

      case DEFINE_METHOD_INSTANCE_NODE:
       { struct method_instance *mi = METH_INSTANCE(node->op1.meth_id);
         sprintf(pos,
           "/* Definition of method instance %s was originally here.*/",
            mi->name);
         pos += strlen(pos);
         return;
       }

      case DEFINE_CONSTRAINT_NODE:
       { struct constraint *con = get_constraint(node->op1.con_id);
         if ( con->attr & NAMED_THING )
           sprintf(pos,
             "/* Definition of constraint %s was originally here.*/",
              con->name);
         else
           sprintf(pos,
             "/* Definition of constraint %d was originally here.*/",
              node->op1.con_id);
         pos += strlen(pos);
         return;
       }

      case DEFINE_BOUNDARY_NODE:
       { struct boundary *bdry = web.boundaries+node->op1.bdry_id;
         if ( bdry->attr & NAMED_THING )
           sprintf(pos,
             "/* Definition of boundary %s was originally here.*/",
              bdry->name);
         else
           sprintf(pos,
             "/* Definition of boundary %d was originally here.*/",
              node->op1.con_id);
         pos += strlen(pos);
         return;
       }

      case DEFINE_EXTRA_NODE:
         ex = EXTRAS(node->op2.eltype)+node->op1.extranum;
         sprintf(pos,"define %s attribute %s %s",
           typenames[node->op2.eltype],ex->name, datatype_name[ex->type]);
         pos += strlen(pos);
         if ( ex->code.start )
         { strcat(pos," function "); pos += strlen(pos);
           /*current_proc_locals[++current_proc_depth] = ex->locals;*/
           exprint_recur(ex->code.root,prec_parent);
           /*current_proc_depth--;*/
         }
         break;

      case DEFINE_EXTRA_INDEX_NODE:
         exprint_recur(node+node->left,prec_parent);
         exprint_recur(node+node->right,prec_parent);
         break; 

      case DEFINE_ARRAY_NODE: 
         sprintf(pos,"define %s %s",globals(node->op1.name_id)->name,
            datatype_name[node->op2.valtype]);
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;

      case DEFINE_FIXED_LOCAL_ARRAY_NODE: 
        { struct global *g = globals(node->op1.name_id);
          struct array *a = g->attr.arrayptr;
          int i;

          sprintf(pos,"define %s %s",g->name,
            datatype_name[node->op2.valtype]);
          pos += strlen(pos);
          for ( i = 0 ; i < a->dim ; i++ )
          { sprintf(pos,"[%d]",a->sizes[i]);
            pos += strlen(pos);
          }
        }
         break;

      case ARRAY_EVAL_NODE:
         exprint_recur(node+node->left,prec_parent);
         break;

      case ARRAY_HEAD_NODE:
         exprint_recur(node+node->left,prec_parent);
         break;

      case ARRAYASSIGN_NODE:
         sprintf(pos,"%s",globals(node->op2.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         switch ( node->op1.assigntype )
         { case ASSIGN_OP: sprintf(pos," := "); break;
           case PLUSASSIGN_OP: sprintf(pos," += "); break;
           case SUBASSIGN_OP: sprintf(pos," -= "); break;
           case MULTASSIGN_OP: sprintf(pos," *= "); break;
           case DIVASSIGN_OP: sprintf(pos," /= "); break;
         } 
         pos += strlen(pos);
         exprint_recur(node+node->right,PREC_ASSIGN);
         break;

      case ARRAYEVAL_NODE:
         sprintf(pos,"%s",globals(node->op2.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;  

      /* initializer array syntax */
      case ARRAYLIST_NODE: 
        exprint_recur(node+node->left,prec_parent);
        *(pos++) = ','; *pos=0;
        exprint_recur(node+node->right,prec_parent);
        break;

      case ARRAYEXPR_NODE:
        *(pos++) = '{'; *pos = 0;
        if ( node->left )
          exprint_recur(node+node->left,prec_parent);
        *(pos++) = '}';
        *pos = 0;
        break;


      /* whole-array syntax */
      case ARRAYIDENT_NODE:
      case FIXED_ARRAY_RVAL_NODE:
         sprintf(pos,"%s",get_name_name(node->op2.name_id,localbase));
         pos += strlen(pos);
         break;

      case ARRAY_ASSIGNOP_ARRAY_NODE:
      case ARRAYEXPR_ASSIGN_NODE:
      case ARRAY_ASSIGNOP_SCALAR_NODE:
      case ARRAY_ASSIGNOP_STRING_NODE:
      case ARRAY_ASSIGNOP_S_X_A_NODE:
      case ARRAY_ASSIGNOP_A_P_A_NODE:
      case ARRAY_ASSIGNOP_A_S_A_NODE:
      case ARRAY_ASSIGNOP_A_X_A_NODE:
         exprint_recur(node+node->left,PREC_ASSIGN);
         if ( !(node->flags & SET_ASSIGNOP) )
         {
           switch ( node->op1.assigntype )
           { case ASSIGN_OP: sprintf(pos," := "); break;
             case PLUSASSIGN_OP: sprintf(pos," += "); break;
             case SUBASSIGN_OP: sprintf(pos," -= "); break;
             case MULTASSIGN_OP: sprintf(pos," *= "); break;
             case DIVASSIGN_OP: sprintf(pos," /= "); break;
           } 
           pos += strlen(pos);
           exprint_recur(node+node->right,PREC_ASSIGN);
         }
         else
         {
           sprintf(pos++,"(");  // kludge for "-" after []     
           exprint_recur(node+node->right,PREC_ASSIGN);
           sprintf(pos++,")");
         }
         break;

      case DOT_NODE:
         binary_print(node,prec_parent,PREC_MUL," * ",PREC_MUL);
         break;

      case ARRAY_LVALUE_INDEXED_NODE:
      case ARRAY_RVALUE_INDEXED_NODE:
         exprint_recur(node+node->left,PREC_INDEX);
         if ( node->right )
           exprint_recur(node+node->right,PREC_INDEX);
     //    *pos = ' '; *(++pos) = 0;  //why would this be needed?
         break;

      case PRINT_ARRAY_LVALUE_NODE:
         sprintf(pos,"print ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;
           
      case ATTRIB_LVALUE_NODE:
      case ARRAY_VERTEX_NORMAL_NODE:
      case ARRAY_EDGE_VECTOR_NODE:
      case ARRAY_FACET_NORMAL_NODE:
         if ( node->left )
         { exprint_recur(node+node->left,prec_parent);
         }
         if ( node->op1.localnum == 0)
           sprintf(pos," %s",get_name_name(node->op2.name_id,localbase));
         else
           sprintf(pos,".%s",get_name_name(node->op2.name_id,localbase));
         pos += strlen(pos);
         break;

      case ARRAY_VERTEX_CONSTRAINT_NORMAL_NODE:
         exprint_recur(node+node->left,prec_parent);
         strcat(pos,".constraint[");
         pos += strlen(pos);
         if ( node[node->right].type == PUSHCONST_NODE )
         { int connum = (int)(node[node->right].op1.real);
           struct constraint *con;
           if ( connum < 1 || connum >= web.maxcon )
             sprintf(pos,"%d",connum);
           else
           { con= get_constraint(connum);
             strcat(pos,con->name);
           }
           pos += strlen(pos);
         }
         else
           exprint_recur(node+node->right,prec_parent);
         strcat(pos,"].normal"); 
         pos += strlen(pos);
         break;

      case ARRAY_RVALUE_NODE:
		  exprint_recur(node+node->left,node->op1.intval=='*'?PREC_MUL:PREC_ADD);
         sprintf(pos," %c ",node->op1.intval);
         pos += strlen(pos);
         exprint_recur(node+node->right,node->op1.intval=='*'?PREC_MUL:PREC_ADD);
         break;

      /* end whole-array syntax */

      case SET_CONSTRAINT_GLOBAL_NODE:
         strcat(pos,"set constraint "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         strcat(pos," global "); pos += strlen(pos);
         break;

      case UNSET_CONSTRAINT_GLOBAL_NODE:
         strcat(pos,"unset constraint "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         strcat(pos," global "); pos += strlen(pos);
         break;

      case SET_CONSTRAINT_NAME_GLOBAL_NODE:
         sprintf(pos,"set constraint %s global",
              get_constraint(node->op3.connum)->name); 
         pos += strlen(pos);   
         break;

      case UNSET_CONSTRAINT_NAME_GLOBAL_NODE:
         sprintf(pos,"unset constraint %s global",
              get_constraint(node->op3.connum)->name); 
         pos += strlen(pos);
         break;

      case RESET_COUNTS_NODE:
         strcat(pos,"reset_counts"); pos += strlen(pos); 
         return;

      case FLUSH_COUNTS_NODE:
         strcat(pos,"flush_counts"); pos += strlen(pos); 
         return;

      case PRINT_PROFILING_NODE:
         strcat(pos,"print profiling"); pos += strlen(pos); 
         return;

      case RESET_PROFILING_NODE:
         strcat(pos,"reset_profiling"); pos += strlen(pos); 
         return;

      case PAUSE_NODE:
         strcat(pos,"pause"); pos += strlen(pos); 
         return;

      case RETURN_NODE:
         strcat(pos,"return "); pos += strlen(pos); 
         if ( node->left )
           exprint_recur(node+node->left,prec_parent);
         return;

      case EVOLVER_VERSION_NODE:
         strcat(pos,"evolver_version"); pos += strlen(pos); 
         return;

      case DATE_AND_TIME_NODE:
         strcat(pos,"date_and_time"); pos += strlen(pos); 
         return;

      case BREAK_NODE:
         if ( node->op2.breakdepth > 1 )
         sprintf(pos,"break %d",node->op2.breakdepth);
         else strcat(pos,"break "); 
         pos += strlen(pos);
         return;

      case CONTINUE_NODE:
         if ( node->op2.breakdepth > 1 )
         sprintf(pos,"continue %d",node->op2.breakdepth);
         else strcat(pos,"continue "); 
         pos += strlen(pos);
         return;

      case HISTORY_NODE:
         strcat(pos,"history "); pos += strlen(pos);
         return;

      case GET_TRANSFORM_EXPR_NODE:
         strcat(pos,"transform_expr "); pos += strlen(pos);
         return;

      case WARNING_MESSAGES_NODE:
         strcat(pos,"warning_messages "); pos += strlen(pos);
         return;

      case DATAFILENAME_NODE:
         strcat(pos,"datafilename "); pos += strlen(pos);
         return;

      case REPEAT_NODE:
         exprint_recur(node+node->right,PREC_ARG);
         nn = node + node->left;
         *pos = ' '; pos++; *pos = 0;
         exprint_recur(nn+nn->left,PREC_ARG);
         return;

      case EXPRLIST_NODE:
         exprint_recur(node+node->left,PREC_ARG);
         if ( node->right )
          { strcat(pos,", "); pos += 2;
            exprint_recur(node+node->right,PREC_ARG);
          }
         return;
     
      case QUOTATION_NODE:
         print_quote(node->op1.string);
         return;

      case SPRINTFHEAD_NODE:
      case PRESPRINTF_NODE:
         sprintf(pos,"sprintf ");
         pos += strlen(pos);
         if ( node->op1.string ) print_quote(node->op1.string);
         else exprint_recur(node+node->left,PREC_ARG);
         return;

      case PREPRINTF_NODE:
      case PRINTFHEAD_NODE:
         sprintf(pos,"printf ");
         pos += strlen(pos);
         if ( node->op1.string ) print_quote(node->op1.string);
         else exprint_recur(node+node->left,PREC_ARG);
         return;

      case ERRPRINTFHEAD_NODE:
         sprintf(pos,"errprintf ");
         pos += strlen(pos);
         if ( node->op1.string ) print_quote(node->op1.string);
         else exprint_recur(node+node->left,PREC_ARG);
         return;

      case BINARY_PRINTFHEAD_NODE:
         sprintf(pos,"binary_printf ");
         pos += strlen(pos);
         if ( node->op1.string ) print_quote(node->op1.string);
         else exprint_recur(node+node->left,PREC_ARG);
         return;

      case PRINTF_NODE:
      case BINARY_PRINTF_NODE:
      case ERRPRINTF_NODE:
      case SPRINTF_NODE:
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,",");
         exprint_recur(node+node->right,PREC_ARG);
         return;

      case STRPRINT_NODE: 
      case PRINT_NODE: 
      case EPRINT_NODE:
         sprintf(pos,"print "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;
    
      case PRINT_LETTER_NODE: 
         sprintf(pos,"print %c ",node->op1.name_id); 
         pos += strlen(pos);
         return;
    
      case PRINT_PROCEDURE_NODE: 
      case PRINT_ARRAY_NODE: 
         sprintf(pos,"print %s ",globals(node->op1.name_id)->name); 
         pos += strlen(pos);
         return;
    
      case PRINT_ARRAYPART_NODE: 
         sprintf(pos,"print ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;
    
      case PRINT_VERTEXNORMAL_NODE:
         sprintf(pos,"print ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,".vertexnormal");
         pos += strlen(pos);
         return;

      case PRINT_ATTR_ARRAY_NODE: 
         sprintf(pos,"print ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_INDEX);
         sprintf(pos,".%s",EXTRAS(node->op2.eltype)[node->op3.extranum].name);
         pos += strlen(pos);
         if ( node->right )
           exprint_recur(node+node->right,PREC_INDEX);
         return;
    
      case PRINT_PERM_PROCEDURE_NODE: 
         sprintf(pos,"print %s ",perm_globals(node->op1.name_id)->name); 
         pos += strlen(pos);
         return;
    
      case EXPRINT_PROCEDURE_NODE: 
         sprintf(pos,"exprint %s ",globals(node->op1.name_id)->name); 
         pos += strlen(pos);
         return;
    
      case ELSE_NODE: /* root of IF */
         exprint_recur(node+node->left,prec_parent); /* IF part */
         if ( node->right )
           { bracket_depth--;
             newline();
             sprintf(pos,"else "); pos += strlen(pos);
             bracket_depth++;
             newline();
             exprint_recur(node+node->right,prec_parent); /* command */
             bracket_depth--;
           }
         else bracket_depth--;
         return;

      case IF_NODE:
         exprint_recur(node+node->left,prec_parent); /* IF part */
         exprint_recur(node+node->right,prec_parent); /* command */
         return;
         
      case IFTEST_NODE:
         sprintf(pos,"if ( "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent); /* expr */
         sprintf(pos," ) then "); pos += strlen(pos);
         bracket_depth++;
         newline();
         return;

      case WHILE_END_NODE:
         exprint_recur(node+node->left,prec_parent); /* test part */
         sprintf(pos," do "); pos += strlen(pos);
         bracket_depth++;
         if ( node->right )
         { newline();
           exprint_recur(node+node->right,prec_parent); /* command */
         } else
         { strcat(pos," ;"); pos += strlen(pos); }
         bracket_depth--;
         return;

      case WHILE_TOP_NODE:
         sprintf(pos,"while ("); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG); /* expr */
         strcat(pos,") "); pos += strlen(pos);
         return;

      case DO_TOP_NODE:
         exprint_recur(node+node->right,prec_parent); /* command */
         return;

      case DO_END_NODE:
         sprintf(pos,"do "); pos += strlen(pos);
         bracket_depth++;
         newline();
         exprint_recur(node+node->left,prec_parent); /* command */
         bracket_depth--;
         newline();
         sprintf(pos," while ("); pos += strlen(pos);
         exprint_recur(node+node->right,PREC_ARG); /* expr */
         strcat(pos,") "); pos += strlen(pos);
         return;

      case FOR_END_NODE:
         sprintf(pos,"for ( "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG); /* FOR_TOP_ */
         sprintf(pos," ) "); pos += strlen(pos);
         bracket_depth++;
         newline();
         if ( node->right )
           exprint_recur(node+node->right,prec_parent); /* command3 */
         else { strcat(pos," ;") ; pos += strlen(pos); } /* empty command3 */
         bracket_depth--;
         newline();
         return;

      case FOR_TOP_NODE:
         exprint_recur(node+node->left,PREC_ARG); /* FOR_HEAD_ */
         sprintf(pos," ; "); pos += strlen(pos);
         exprint_recur(node+node->right,prec_parent); /* command2 */
         return;

      case FOR_HEAD_NODE:
         exprint_recur(node+node->left,PREC_ARG); /* FOR_ENTRY */
         sprintf(pos," ; "); pos += strlen(pos);
         exprint_recur(node+node->right,PREC_ARG); /* expr */
         return;

      case FOR_ENTRY_NODE: 
         exprint_recur(node+node->left,prec_parent); /* command1 */
         return;

      case REDIRECT_NODE:
         if ( node->left ) 
           { strcat(pos," >> "); pos += strlen(pos);
             exprint_recur(node+node->left,prec_parent); /* command */
           }
         else
           sprintf(pos," >> \"%s\" ",node->op1.string); pos += strlen(pos);
         return;

      case REDIRECTOVER_NODE:
         if ( node->left ) 
           { strcat(pos," >>> "); pos += strlen(pos);
             exprint_recur(node+node->left,prec_parent); /* command */
           }
         else
           sprintf(pos," >>> \"%s\" ",node->op1.string); pos += strlen(pos);
         return;

       case REDIRECT_ERR_NODE:
         if ( node->left ) 
           { strcat(pos," >>2 "); pos += strlen(pos);
             exprint_recur(node+node->left,prec_parent); /* command */
           }
         else
           sprintf(pos," >>2 \"%s\" ",node->op1.string); pos += strlen(pos);
         return;

      case REDIRECTOVER_ERR_NODE:
         if ( node->left ) 
           { strcat(pos," >>>2 "); pos += strlen(pos);
             exprint_recur(node+node->left,prec_parent); /* command */
           }
         else
           sprintf(pos," >>>2 \"%s\" ",node->op1.string); pos += strlen(pos);
         return;


      case PIPE_NODE:
         if ( node->left ) 
           { strcat(pos," | "); pos += strlen(pos);
             exprint_recur(node+node->left,prec_parent); /* command */
           }
         else
           sprintf(pos," | \"%s\" ",node->op1.string); pos += strlen(pos);
         return;

      case PIPE_END_NODE:
      case REDIRECT_END_NODE:
         exprint_recur(node+node->right,prec_parent); /* command */
         exprint_recur(node+node->left,prec_parent); /* pipe */
         return;

      case SINGLE_REDEFD_NODE:
         sprintf(pos,"%c ",node->op1.letter); 
         pos += strlen(pos);
         return;

      case SINGLE_LETTER_NODE:
         if ( single_redefine[node->op1.letter].start )
            sprintf(pos,"'%c' ",node->op1.letter);
         else sprintf(pos,"%c ",node->op1.letter); 
         pos += strlen(pos);
         return;

    /* toggles */ case QUIETGO_NODE: case QUIETLOAD_NODE: case SLICE_VIEW_NODE:
    case CLIP_VIEW_NODE: case STAR_FINAGLING_NODE: case FORCE_DELETION_NODE:
    case KUSNER_NODE: case ESTIMATE_NODE: case DETURCK_NODE: case HOMOTHETY_NODE:
    case SQGAUSS_NODE: case AUTOPOP_NODE: case QUIET_NODE:
    case OLD_AREA_NODE: case APPROX_CURV_NODE: case RUNGE_KUTTA_NODE: 
    case CHECK_INCREASE_NODE:  case DEBUG_NODE: case MEAN_CURV_NODE: case RIBIERE_CG_NODE:
    case DIFFUSION_NODE: case GRAVITY_NODE: case CONJ_GRAD_NODE: case TRANSFORMS_NODE:
    case CONF_EDGE_SQCURV_NODE: case EFFECTIVE_AREA_NODE: 
    case RAW_CELLS_NODE: case CONNECTED_CELLS_NODE: case CLIPPED_CELLS_NODE:
    case THICKEN_NODE: case SHOW_INNER_NODE: case SHOW_OUTER_NODE: case COLORMAP_NODE: 
    case HESSIAN_DIFF_NODE: case POST_PROJECT_NODE: case MEAN_CURV_INT_NODE:
    case NORMAL_CURVATURE_NODE: case DIV_NORMAL_CURVATURE_NODE:
    case SHADING_NODE:  case FACET_COLORS_NODE: case BOUNDARY_CURVATURE_NODE:
    case NORMAL_MOTION_NODE: case PINNING_NODE: case VIEW_4D_NODE: case MEMDEBUG_NODE:
    case METRIC_CONVERSION_NODE: case AUTORECALC_NODE: case GV_BINARY_NODE:
    case SELF_SIMILAR_NODE: case AUTODISPLAY_NODE: case FORCE_POS_DEF_NODE:
    case ASSUME_ORIENTED_NODE: case HESSIAN_QUIET_NODE: case JIGGLE_TOGGLE_NODE:
    case HESSIAN_NORMAL_NODE: case YSMP_NODE: case BUNCH_KAUFMAN_NODE:
    case QUANTITIES_ONLY_NODE: case LINEAR_METRIC_NODE: case DETORUS_STICKY_NODE:
    case SQUARED_GRADIENT_NODE: case H_INVERSE_METRIC_NODE: case MKL_NODE:
    case HESSIAN_DOUBLE_NORMAL_NODE: case INTERP_BDRY_PARAM_NODE: 
    case HESSIAN_NORMAL_ONE_NODE: case PSCOLORFLAG_NODE: case GRIDFLAG_NODE:
    case SEPTUM_FLAG_NODE: case BOX_FLAG_NODE: case SHOW_ALL_EDGES_NODE:
    case CROSSINGFLAG_NODE: case LABELFLAG_NODE: case SHOW_ALL_QUANTITIES_NODE:
    case HESSIAN_NORMAL_PERP_NODE: case HESSIAN_SPECIAL_NORMAL_NODE: case ITDEBUG_NODE:
    case METIS_FACTOR_NODE: case VOLGRADS_EVERY_NODE: case ZENER_DRAG_NODE: 
    case BACKCULL_NODE: case INTERP_NORMALS_NODE: case TORUS_FILLED_NODE: case VERBOSE_NODE:
    case AMBIENT_PRESSURE_NODE: case DIRICHLET_MODE_NODE: case SOBOLEV_MODE_NODE:
    case KRAYNIKPOPVERTEX_FLAG_NODE: case FUNCTION_QUANTITY_SPARSE_NODE:
    case KRAYNIKPOPEDGE_FLAG_NODE: case VISIBILITY_TEST_NODE: case SPARSE_CONSTRAINTS_NODE:
    case K_ALTITUDE_FLAG_NODE: case FORCE_EDGESWAP_NODE:
    case BLAS_FLAG_NODE: case AUGMENTED_HESSIAN_NODE: case BREAK_AFTER_WARNING_NODE:
    case RGB_COLORS_FLAG_NODE:  case CIRCULAR_ARC_DRAW_NODE: case BEZIER_BASIS_NODE:
    case SMOOTH_GRAPH_NODE: case MPI_DEBUG_NODE: case POP_DISJOIN_NODE:
    case POP_TO_EDGE_NODE: case POP_TO_FACE_NODE: case POP_ENJOIN_NODE:
    case FULL_BOUNDING_BOX_NODE: case BIG_ENDIAN_NODE: case LITTLE_ENDIAN_NODE:
    case AUTOPOP_QUARTIC_NODE: case IMMEDIATE_AUTOPOP_NODE: case ROTATE_LIGHTS_NODE:
    case VIEW_TRANSFORMS_USE_UNIQUE_NODE: case PS_CMYKFLAG_NODE: case BREAK_ON_WARNING_NODE:
         sprintf(pos,"%s %s ",keywordname(node->type),
             node->op1.toggle_state==ON_?"ON":"OFF"); 
         pos += strlen(pos);
         break;

    case OPTIMIZE_NODE: 
         sprintf(pos,"optimize %s ",
          node->op1.toggle_state==ON_?"ON":"OFF"); pos += strlen(pos);
         break;
   
    case AUTOCHOP_NODE: 
            sprintf(pos,"autochop %s ",
          node->op1.toggle_state==ON_?"ON":"OFF"); pos += strlen(pos);
         break;
    
    case LOGFILE_TOGGLE_NODE:
         sprintf(pos,"logfile %s ",
          node->op1.toggle_state==ON_?"ON":"OFF"); pos += strlen(pos);
         break;

    case KEYLOGFILE_TOGGLE_NODE:
         sprintf(pos,"keylogfile %s ",
          node->op1.toggle_state==ON_?"ON":"OFF"); pos += strlen(pos);
         break;

    case GEOMVIEW_TOGGLE_NODE:
         sprintf(pos,"geomview %s ",
          node->op1.toggle_state==ON_?"ON":"OFF"); pos += strlen(pos);
         break;

    case GEOMPIPE_TOGGLE_NODE:
         sprintf(pos,"geompipe %s ",
          node->op1.toggle_state==ON_?"ON":"OFF"); pos += strlen(pos);
         break;

    case GEOMPIPE_NODE:
         sprintf(pos,"geompipe "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;

    case POSTSCRIPT_NODE:
         sprintf(pos,"postscript "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;

    case BINARY_OFF_FILE_NODE:
         sprintf(pos,"binary_off_file "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;

    case OOGLFILE_NODE:
         sprintf(pos,"ooglfile "); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         break;

    case TOGGLEVALUE_NODE:
         sprintf(pos,"(%s) ",keywordname(node->op1.toggle_id)); 
         pos += strlen(pos);
         break;
    
    case GET_INTERNAL_NODE:
         sprintf(pos,"%s",keywordname(node->op1.name_id));
         pos += strlen(pos);
         break;

    case PROCEDURE_NODE:
         sprintf(pos,"%s ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         break;

    case PERM_PROCEDURE_NODE:
         sprintf(pos,"%s ",perm_globals(node->op1.name_id)->name);
         pos += strlen(pos);
         break;

    case FIX_PARAMETER_NODE:
          sprintf(pos,"fix %s",globals(node->op1.name_id)->name);
          pos += strlen(pos);
          break;

    case UNFIX_PARAMETER_NODE:
          sprintf(pos,"unfix %s",globals(node->op1.name_id)->name);
          pos += strlen(pos);
          break;

    case FIX_QUANTITY_NODE:
          sprintf(pos,"fix %s",GEN_QUANT(node->op1.quant_id)->name);
          pos += strlen(pos);
          break;

    case UNFIX_QUANTITY_NODE:
          sprintf(pos,"unfix %s",GEN_QUANT(node->op1.quant_id)->name);
          pos += strlen(pos);
          break;

    case SET_Q_FIXED_NODE:
          sprintf(pos,"set %s fixed",GEN_QUANT(node->op1.quant_id)->name);
          pos += strlen(pos);
          break;

    case SET_Q_ENERGY_NODE:
          sprintf(pos,"set %s energy",GEN_QUANT(node->op1.quant_id)->name);
          pos += strlen(pos);
          break;

    case SET_Q_INFO_NODE:
          sprintf(pos,"set %s info_only",GEN_QUANT(node->op1.quant_id)->name);
          pos += strlen(pos);
          break;

    case SET_Q_CONSERVED_NODE:
          sprintf(pos,"set %s conserved",GEN_QUANT(node->op1.quant_id)->name);
          pos += strlen(pos);
          break;

    case SET_INTERNAL_NODE:
         sprintf(pos,"%s %s ",keywordname(node->op1.name_id),
           assign_symbol(node->op2.assigntype));
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case VIEW_MATRIX_LVALUE_NODE:
         sprintf(pos,"view_matrix["); pos += strlen("view_matrix[");
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,"]["); pos += 2;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,"]"); pos += 1;
         break;

    case SET_VIEW_MATRIX_NODE:
         exprint_recur(node+node->left,PREC_ASSIGN);
         sprintf(pos," := "); pos += 4;
         exprint_recur(node+node->right,PREC_ASSIGN);
         break;

    case SET_SCALE_NODE:
          sprintf(pos,"m "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_OPTIMIZE_NODE:
          sprintf(pos,"optimize "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_GAP_CONSTANT_NODE:
          sprintf(pos,"gap_constant := "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SKINNY_NODE:
          sprintf(pos,"K "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case TORDUP_NODE:
          sprintf(pos,"y "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_MODEL_NODE:
          sprintf(pos,"M "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case INVOKE_P_MENU_NODE:
          sprintf(pos,"P "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_GRAVITY_NODE:
          switch ( node->op1.assigntype )
          { case ASSIGN_OP: sprintf(pos,"G "); break;
            case PLUSASSIGN_OP: sprintf(pos,"gravity += "); break;
            case SUBASSIGN_OP: sprintf(pos,"gravity -= "); break;
            case MULTASSIGN_OP: sprintf(pos,"gravity *= "); break;
            case DIVASSIGN_OP: sprintf(pos,"gravity /= "); break;
          } 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_DIFFUSION_NODE:
          sprintf(pos,"diffusion := "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_THICKEN_NODE:
          sprintf(pos,"thicken := "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_AUTOCHOP_NODE:
          sprintf(pos,"autochop := "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_AMBIENT_PRESSURE_NODE:
          sprintf(pos,"p "); pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case ZOOM_NODE:
          sprintf(pos,"zoom "); pos+=strlen(pos);
          if ( node->left )
            { exprint_recur(node+node->left,PREC_ASSIGN);
              exprint_recur(node+node->right,PREC_ASSIGN);
            }
          break;

    case CHDIR_NODE:
         sprintf(pos,"chdir ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

   case SYSTEM_NODE:
         sprintf(pos,"system ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

   case EXEC_NODE: case PARALLEL_EXEC_NODE:
         sprintf(pos,"exec ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case READ_NODE:
         sprintf(pos,"read ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case LOAD_NODE:
         sprintf(pos,"load ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case PERMLOAD_NODE:
         sprintf(pos,"permload ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case ADDLOAD_NODE:
         sprintf(pos,"addload ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case REPLACE_LOAD_NODE:
         sprintf(pos,"replace_load ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case SHOW_TRANS_NODE:
         sprintf(pos,"show_trans ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case TRANSFORM_EXPR_NODE:
         sprintf(pos,"transform_expr ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

   case GEOMVIEW_NODE:
         sprintf(pos,"geomview ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;
 
    case TASK_EXEC_NODE:
         sprintf(pos,"%s ",keywordname(node->type));
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         *pos = ','; pos++; *pos = 0;
         exprint_recur(node+node->right,PREC_ARG);
         break;

      case VIEW_TRANSFORM_PARITY_NODE:
         strcat(pos,"view_transform_parity["); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,"]"); pos++;
         break;

      case VIEW_TRANSFORM_SWAP_COLORS_NODE:
         strcat(pos,"view_transform_swap_colors["); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,"]"); pos++;
         break;

      case VIEW_TRANSFORMS_NOP_NODE:
         strcat(pos,"view_transforms["); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,"]["); pos += 2;
         exprint_recur(node+node->right,PREC_ARG);
         break;

      case VIEW_TRANSFORMS_ELEMENT_NODE:
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,"]["); pos += 2;
         exprint_recur(node+node->right,PREC_ARG);
         *(pos++) = ']'; *pos = 0;
         break;

    case IS_DEFINED_NODE:
         sprintf(pos,"is_defined(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         *(pos++) = ')'; *pos = 0;
         break;

    case DUMP_NODE:
         sprintf(pos,"dump ");
         pos += strlen(pos);
         if ( node->left )
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

    case SET_COLORMAP_NODE:
         sprintf(pos,"colormap := "); 
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

      case SHOW_VOL_NODE: sprintf(pos,"show_vol "); pos+=strlen(pos); break; 
      case CHECK_NODE:sprintf(pos,"check "); pos+=strlen(pos); break; 
      case LONG_JIGGLE_NODE: sprintf(pos,"long_jiggle "); pos+=strlen(pos); break;
      case RAW_VERAVG_NODE:sprintf(pos,"rawv "); pos+=strlen(pos); break;
      case STABILITY_TEST_NODE:sprintf(pos,"stability_test "); pos+=strlen(pos); break; 
      case UTEST_NODE: sprintf(pos,"utest "); pos+=strlen(pos); break;
      case GO_NODE: sprintf(pos,"g "); pos+=strlen(pos); break;
      case SHELL_NODE: sprintf(pos,"shell "); pos+=strlen(pos); break;
      case ALICE_NODE:   sprintf(pos,"alice "); pos+=strlen(pos); break; 
      case RECALC_NODE: sprintf(pos,"recalc "); pos+=strlen(pos); break;
      case COUNTS_NODE: sprintf(pos,"counts "); pos+=strlen(pos); break;
      case RAWEST_VERAVG_NODE:sprintf(pos,"rawestv "); pos+=strlen(pos); break;
      case EXTRAPOLATE_NODE:sprintf(pos,"extrapolate "); pos+=strlen(pos); break; 
      case LINEAR_NODE: sprintf(pos,"linear "); pos+=strlen(pos); break;
      case QUADRATIC_NODE:sprintf(pos,"quadratic "); pos+=strlen(pos); break; 
      case REBODY_NODE:sprintf(pos,"rebody "); pos+=strlen(pos); break;
      case HESSIAN_NODE: sprintf(pos,"hessian "); pos+=strlen(pos); break;
      case SHOWQ_NODE: sprintf(pos,"simplex_to_fe"); pos+=strlen(pos); break;
      case CLOSE_SHOW_NODE: sprintf(pos,"showq "); pos+=strlen(pos); break;
      case HESSIAN_MENU_NODE:sprintf(pos,"hessian_menu "); pos+=strlen(pos); break;
      case DIRICHLET_NODE: sprintf(pos,"dirichlet "); pos+=strlen(pos); break;
      case SOBOLEV_NODE: sprintf(pos,"sobolev "); pos+=strlen(pos); break;
      case REORDER_STORAGE_NODE:sprintf(pos,"reorder_storage "); pos+=strlen(pos); break;
      case DIRICHLET_SEEK_NODE: sprintf(pos,"dirichlet_seek "); pos+=strlen(pos); break;
      case SOBOLEV_SEEK_NODE: sprintf(pos,"sobolev_seek "); pos+=strlen(pos); break;
      case CONVERT_TO_QUANTS_NODE:sprintf(pos,"convert_to_quantities "); pos+=strlen(pos); break;
      case RENUMBER_ALL_NODE: sprintf(pos,"renumber_all "); pos+=strlen(pos); break;
      case DUMP_MEMLIST_NODE: sprintf(pos,"dump_memlist "); pos+=strlen(pos); break;
      case FREE_DISCARDS_NODE:sprintf(pos,"free_discards "); pos+=strlen(pos); break;
      case REPARTITION_NODE: sprintf(pos,"repartition "); pos+=strlen(pos); break;
      case SUBCOMMAND_NODE: sprintf(pos,"subcommand "); pos+=strlen(pos); break;
      case ABORT_NODE: sprintf(pos,"abort "); pos+=strlen(pos); break;
      case DETORUS_NODE:sprintf(pos,"detorus "); pos+=strlen(pos); break;
      case MAKE_THREAD_LISTS_NODE:sprintf(pos,"make_thread_lists "); pos+=strlen(pos); break;
      case SIMPLEX_TO_FE_NODE: sprintf(pos,"simplex_to_fe"); pos+=strlen(pos); break;
             
      case BURCHARD_NODE:
         sprintf(pos,"burchard %d ",node->op1.maxsteps); 
         pos+=strlen(pos);
         break;
             
      case SHOW_END_NODE:
         if ( node[node->left].type == SHOW_NODE )
           sprintf(pos,"show_expr ");  /* prevent extraneous shows from dump */
         else sprintf(pos,"show_expr ");
         pos += strlen(pos);
         exprint_recur(node+node->right,PREC_ASSIGN);
         break;

      case SET_PROC_END_NODE:
       { struct global *g = globals(node[node->left].op1.name_id);
         if ( node[node->left].type == REDEFINE_SINGLE_NODE )
         { struct treenode *nnode = node+node->left;
           sprintf(pos,"%c :::= ",nnode->op1.letter);
           pos += strlen(pos);
           exprint_recur(nnode+nnode->op2.jumpsize,0);
           break;
         }
         else if ( g->flags & PERMANENT )
          sprintf(pos,"%s ::= ",g->name);
         else  sprintf(pos,"%s := ",g->name);
         pos += strlen(pos);
         if ( node[node->right].type != COMMAND_BLOCK_NODE )
         { strcat(pos,"{"); pos++; } 
         current_proc_locals[++current_proc_depth] = g->value.proc.locals;
         exprint_recur(node+node->right,prec_parent);
         current_proc_depth--;
         if ( node[node->right].type != COMMAND_BLOCK_NODE )
         { strcat(pos,"}"); pos++; } 
         break;
       }
      case SET_PERM_PROC_END_NODE:
         sprintf(pos,"%s ::= ",perm_globals(node[node->left].op1.name_id)->name);
         pos += strlen(pos);
         if ( node[node->right].type != COMMAND_BLOCK_NODE )
         { strcat(pos,"{"); pos++; } 
         exprint_recur(node+node->right,prec_parent);
         if ( node[node->right].type != COMMAND_BLOCK_NODE )
         { strcat(pos,"}"); pos++; } 
         break;

      case ARGLIST_NODE:
       { struct global *g;
         if ( node->op1.name_id == 0 ) break; /* empty list */
         g = globals(node->op1.name_id);
         if ( node->left )
         { exprint_recur(node+node->left,PREC_ARG); /* arglist */
           strcat(pos,","); pos++;
         }
         sprintf(pos,"%s %s",datatype_name[node->op3.argtype],g->name);
         pos += strlen(pos);
         break;
       }

      case FUNCTION_CALL_NODE:
       { struct global *g = globals(node->op1.name_id);
         sprintf(pos,"%s(",g->name);
         pos += strlen(pos);  
         if ( node->left )
           exprint_recur(node+node->left,PREC_ARG); /* arglist */
         strcat(pos,")"); pos++;
         break;
       }

      case FUNCTION_CALL_RETURN_NODE:
         exprint_recur(node+node->left,prec_parent); /* FUNCTION_CALL_ */
         break;

      case FUNCTION_START_NODE: break;
      case FUNCTION_DEF_START_NODE: break;
      case FUNCTION_PROTO_START_NODE: break;

      case FUNCTION_HEAD_NODE:
         exprint_recur(node+node->right,prec_parent); /* arglist */
         break;

      case SET_FUNCTION_NODE:
      case FUNCTION_PROTO_NODE:
       { struct global *g = globals(node[node->left].op1.name_id);
         sprintf(pos,"function %s %s (",datatype_name[node->op4.ret_type],
                g->name);
         pos += strlen(pos);
         current_proc_locals[++current_proc_depth] = g->value.proc.locals;
         exprint_recur(node+node->left,PREC_ARG); /* arglist */
         if ( node->type == SET_FUNCTION_NODE )
         { strcat(pos,")\n"); pos += strlen(pos);
           exprint_recur(node+node->right,prec_parent); /* body */
         }
         else { strcat(pos,");\n"); pos += strlen(pos); }
         current_proc_depth--;
         break;
       }
      case PROCEDURE_CALL_NODE:
         sprintf(pos,"%s(",globals(node->op1.name_id)->name);
         pos += strlen(pos); 
         if ( node->left ) 
            exprint_recur(node+node->left,prec_parent); /* arglist */
         strcat(pos,")"); pos++;
         break;
      case PROCEDURE_CALL_RETURN_NODE:
         exprint_recur(node+node->left,prec_parent); /* PROCEDURE_CALL_ */
         break;

      case PROCEDURE_START_NODE: break;
      case PROCEDURE_DEF_START_NODE: break;
      case PROCEDURE_PROTO_START_NODE: break;

      case PROCEDURE_HEAD_NODE:
         exprint_recur(node+node->right,PREC_ARG); /* arglist */
         break;

      case SET_ARGSPROC_NODE:
      case PROCEDURE_PROTO_NODE:
       { struct global *g = globals(node[node->left].op1.name_id);
         sprintf(pos,"procedure %s (",g->name);
         pos += strlen(pos);
         current_proc_locals[++current_proc_depth] = g->value.proc.locals;
         exprint_recur(node+node->left,PREC_ARG); /* arglist */
         if ( node->type == SET_ARGSPROC_NODE )
         { strcat(pos,")\n"); pos += strlen(pos);
           exprint_recur(node+node->right,prec_parent); /* body */
         }
         else { strcat(pos,");\n"); pos += strlen(pos); }
         current_proc_depth--;
         break;
       }

      case DEFINE_IDENT_NODE:
         sprintf(pos,"define %s %s",globals(node->op1.name_id)->name,
              datatype_name[node->op2.valtype]);
         pos += strlen(pos);
         break;

    case SET_DELTA_NODE:
          sprintf(pos,"%s.pdelta",globals(node->op1.name_id)->name);
          pos += strlen(pos);
          switch ( node->op2.assigntype )
          { case ASSIGN_OP: sprintf(pos," := "); break;
            case PLUSASSIGN_OP: sprintf(pos," += "); break;
            case SUBASSIGN_OP: sprintf(pos," -= "); break;
            case MULTASSIGN_OP: sprintf(pos," *= "); break;
            case DIVASSIGN_OP: sprintf(pos," /= "); break;
          } 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_PARAM_SCALE_NODE:
          sprintf(pos,"%s.pscale",globals(node->op1.name_id)->name);
          pos += strlen(pos);
          switch ( node->op2.assigntype )
          { case ASSIGN_OP: sprintf(pos," := "); break;
            case PLUSASSIGN_OP: sprintf(pos," += "); break;
            case SUBASSIGN_OP: sprintf(pos," -= "); break;
            case MULTASSIGN_OP: sprintf(pos," *= "); break;
            case DIVASSIGN_OP: sprintf(pos," /= "); break;
          } 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_ON_ASSIGN_CALL_NODE:
      { struct global *g = globals(node->op1.name_id);
        sprintf(pos,"%s.on_assign_call := %s",g->name,
             globals(g->attr.varstuff.on_assign_call)->name);
        pos += strlen(pos);
        break;
      }

      case SET_GLOBAL_NODE: case SET_SGLOBAL_NODE:
         { struct global *g = globals(node->op1.name_id);
           if ( g->flags & PERMANENT )
           sprintf(pos,"%s ::= ",g->name);
           else sprintf(pos,"%s := ",g->name);
           pos += strlen(pos);
           exprint_recur(node+node->left,PREC_ASSIGN);
           break;
         }
      case SET_PERM_GLOBAL_NODE: case SET_PERM_SGLOBAL_NODE:
         sprintf(pos,"%s ::= ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

      case SET_NO_DUMP_NODE:
         if ( node->op2.intval )
           sprintf(pos,"%s.no_dump on ",globals(node->op1.name_id)->name);
         else 
            sprintf(pos,"%s.no_dump off ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         break;

      case PLUSASSIGN_NODE:
         sprintf(pos,"%s += ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

      case SUBASSIGN_NODE:
         sprintf(pos,"%s -= ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

      case MULTASSIGN_NODE:
         sprintf(pos,"%s *= ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

      case DIVASSIGN_NODE:
         sprintf(pos,"%s /= ",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         break;

      case PRE_INCREMENT_NODE:
         sprintf(pos,"%s%s",node->op2.assigntype==PLUSPLUS_OP ? " ++" : " --",
            globals(node->op1.name_id)->name);
         pos += strlen(pos);
         break;

      case POST_INCREMENT_NODE:
         sprintf(pos,"%s%s",globals(node->op1.name_id)->name,
             node->op2.assigntype==PLUSPLUS_OP ? "++ " : "-- ");
         pos += strlen(pos);
         break;


      case SIZEOF_ATTR_NODE:
         sprintf(pos,"sizeof(%s)",
           EXTRAS(node->op2.eltype)[node->op1.extranum].name); 
         pos += strlen(pos);
         break;

      case SIZEOF_ARRAY_NODE:
         sprintf(pos,"sizeof(%s)",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         break;

      case SIZEOF_STRING_NODE:
         sprintf(pos,"sizeof(");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         strcat(pos,")");
         pos++;
         break;

      case LAGRANGE_NODE:
         sprintf(pos,"lagrange "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ASSIGN);
         return;

      case LANCZOS_NODE: 
         sprintf(pos,"lanczos "); pos += strlen(pos);
         if ( node->right )
         { strcat(pos,"("); pos++;
           exprint_recur(node+node->left,PREC_ARG);
           strcat(pos,","); pos++;
           exprint_recur(node+node->right,PREC_ARG);
           strcat(pos,")"); pos++;
         }
         else exprint_recur(node+node->left,PREC_ASSIGN);
         return;

       case EIGENPROBE_NODE: 
         sprintf(pos,"eigenprobe "); pos += strlen(pos);
         if ( node->right )
         { strcat(pos,"("); pos++;
           exprint_recur(node+node->left,PREC_ARG);
           strcat(pos,","); pos++;
           exprint_recur(node+node->right,PREC_ARG);
           strcat(pos,")"); pos++;
         }
         else exprint_recur(node+node->left,PREC_ASSIGN);
         return;

      case RITZ_NODE:
         sprintf(pos,"ritz("); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos,")"); pos++;
         return;

      case GET_TORUS_PERIODS_NODE:
         sprintf(pos,"torus_periods"); pos += strlen(pos);
         strcat(pos++,"[");
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,"]["); pos+=2;
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,"]");
         return;

      case GET_INVERSE_PERIODS_NODE:
         sprintf(pos,"inverse_periods"); pos += strlen(pos);
         strcat(pos++,"[");
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,"]["); pos+=2;
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,"]");
         return;

      case HESSIAN_SADDLE_NODE:
         sprintf(pos,"saddle "); pos += strlen(pos);
         if ( node->left )
           exprint_recur(node+node->left,PREC_ARG);
         return;
   
      case HESSIAN_SEEK_NODE:
         sprintf(pos,"hessian_seek "); pos += strlen(pos);
         if ( node->left )
           exprint_recur(node+node->left,PREC_ARG);
         return;

      case MOVE_NODE:
         sprintf(pos,"move "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

     case AREAWEED_NODE:
         sprintf(pos,"areaweed "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

     case EDGEWEED_NODE:
        sprintf(pos,"edgeweed "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case METIS_NODE:
        sprintf(pos,"metis "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case METIS_READJUST_NODE:
        sprintf(pos,"metis_readjust "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case KMETIS_NODE:
        sprintf(pos,"kmetis "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case BODY_METIS_NODE:
        sprintf(pos,"body-metis "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case NOTCH_NODE:
        sprintf(pos,"notch "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case EDGEDIVIDE_NODE:
         sprintf(pos,"edge_divide "); pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case OMETIS_NODE:
         sprintf(pos,"ometis "); pos += strlen(pos);
         if ( node->left )
           exprint_recur(node+node->left,PREC_ARG);
         return;

      case JIGGLE_NODE:
         strcat(pos,"j "); pos += 2;
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case QUIT_NODE:
         strcat(pos,"exit "); pos += 5;
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case LIST_PROCS_NODE:
         sprintf(pos,"list procedures "); pos+=strlen(pos);
         break;
             
      case LIST_ATTRIBUTES_NODE:
         sprintf(pos,"list attributes "); pos+=strlen(pos);
         break;

      case LIST_QUANTITY_NODE:
         sprintf(pos,"list quantity %s",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         break;

      case LIST_METHOD_INSTANCE_NODE:
         sprintf(pos,"list method_instance %s",
              METH_INSTANCE(node->op1.meth_id)->name);
         pos += strlen(pos);
         break;
             
      case LIST_CONSTRAINT_NODE:
         {
           if ( node->op1.con_id > 0 )
           { sprintf(pos,"list constraint %s",
                get_constraint(node->op1.con_id)->name);
             pos += strlen(pos);
           }
           else
           { sprintf(pos,"list constraint ");
             pos += strlen(pos);
             exprint_recur(node+node->left,PREC_ARG);
           }
         }    
         break;

      case LIST_BOUNDARY_NODE:
         { 
           if ( node->op1.bdry_id > 0 )
           { sprintf(pos,"list constraint %s",
                web.boundaries[node->op1.bdry_id].name);
             pos += strlen(pos);
           }
           else
           { sprintf(pos,"list boundary ");
             pos += strlen(pos);
             exprint_recur(node+node->left,PREC_ARG);
           }
         }    
         break;

      case TOPINFO_NODE:
         sprintf(pos,"list topinfo "); pos+=strlen(pos);
         break;
             
      case BOTTOMINFO_NODE:
         sprintf(pos,"list bottominfo "); pos+=strlen(pos);
         break;
             
      case AGGREGATE_END_NODE:
         exprint_recur(node+node->right,prec_parent);
         return;

      case LIST_NODE: 
        sprintf(pos,"list ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         return;

      case DELETE_NODE: 
        sprintf(pos,"delete ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case REFINE_NODE: 
        sprintf(pos,"refine ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case DISSOLVE_NODE: 
        sprintf(pos,"dissolve ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case POP_NODE:
        sprintf(pos,"pop ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case FIX_NODE: 
        sprintf(pos,"fix ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case UNFIX_NODE: 
         sprintf(pos,"unfix ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case EDGESWAP_NODE: 
         sprintf(pos,"edgeswap ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case VERTEX_AVERAGE_NODE:
         sprintf(pos,"vertex_average ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case RAW_VERTEX_AVERAGE_NODE: 
        sprintf(pos,"raw_vertex_average ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case RAWEST_VERTEX_AVERAGE_NODE:
        sprintf(pos,"rawest_vertex_average ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case EQUIANGULATE_NODE: 
         sprintf(pos,"equiangulate ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case POP_EDGE_TO_TRI_NODE: 
         sprintf(pos,"pop_edge_to_tri ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case POP_TRI_TO_EDGE_NODE:
         sprintf(pos,"pop_tri_to_edge ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case POP_QUAD_TO_QUAD_NODE: 
         sprintf(pos,"pop_quad_to_quad ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case T1_EDGESWAP_NODE:
         sprintf(pos,"t1_edgeswap ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case REVERSE_ORIENTATION_NODE:
         sprintf(pos,"reverse_orientation ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

     case WHEREAMI_COMMAND_NODE:
         sprintf(pos,"whereami ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;

      case SINGLE_ELEMENT_INIT_NODE:
         return;

      case SINGLE_ELEMENT_NODE:
         exprint_recur(node+node->left,prec_parent);
         if ( node[node->left].type == INDEXED_SUBTYPE_NODE ||
           node[node->left].type == INDEXED_ELEMENT_NODE )
           if ( node[node->left].op5.string )
             { sprintf(pos," %s ",node[node->left].op5.string);
               pos += strlen(pos);
             }      
         return;

      case GET_VERTEXNORMAL_NODE:
         strcat(pos,"vertexnormal"); pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         return;


      case INDEXED_COORD_NODE:
         strcat(pos,"x"); pos += 1;
         exprint_recur(node+node->left,PREC_INDEX);
         return;

      case INDEXED_ELEMENT_NODE:
         switch ( node->op1.eltype )
         { case VERTEX: strcat(pos,"vertex["); break;
           case EDGE: strcat(pos,"edge["); break;
           case FACET: strcat(pos,"facet["); break;
           case BODY: strcat(pos,"body["); break;
           case FACETEDGE: strcat(pos,"facetedge["); break;
         }
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,"]");
         return;

      case INDEXED_SUBTYPE_NODE:
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,".%s[",typenames[node->op1.eltype]);
         pos += strlen(pos);
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos,"]");
         pos += strlen(pos);
         return;

      case SELF_ELEMENT_NODE:
         strcat(pos,"self"); pos+=4; break;

      case SINGLE_ELEMENT_EXPR_NODE:
         exprint_recur(node+node->left,prec_parent);
         break;

      case SYMBOL_ELEMENT_NODE:
         sprintf(pos,"%s", node->op5.string);
         pos += strlen(pos); break;

      case SET_HIT_PARTNER_NODE:
         set_print(node,"set","hit_partner",PREC_ARG); break;

      case SET_NO_REFINE_NODE:
         set_print(node,"set","no_refine",prec_parent); break;

      case UNSET_NO_REFINE_NODE:
         set_print(node,"unset","no_refine",prec_parent); break;

      case SET_NO_TRANSFORM_NODE:
         set_print(node,"set","no_transform",prec_parent); break;

      case UNSET_NO_TRANSFORM_NODE:
         set_print(node,"unset","no_transform",prec_parent); break;

      case SET_CENTEROFMASS_NODE:
         set_print(node, "set","centerofmass",prec_parent); break;

      case UNSET_CENTEROFMASS_NODE:
         set_print(node,"unset","centerofmass",prec_parent); break;

      case SET_NO_DISPLAY_NODE:
         set_print(node,"set","no_display",prec_parent); break;

      case SET_PHASE_NODE:
         set_print(node,"set","phase",prec_parent); break;

      case SET_BARE_NODE:
         set_print(node,"set","bare",prec_parent); break;

      case SET_DENSITY_NODE:
         set_print(node,"set","density",prec_parent); break;

      case SET_VOLCONST_NODE:
         set_print(node,"set","volconst",prec_parent); break;

      case SET_TARGET_NODE:
         set_print(node,"set","target",prec_parent); break;

      case SET_PRESSURE_NODE:
         set_print(node,"set","pressure",prec_parent); break;

      case SET_OPACITY_NODE:
         set_print(node,"set","opacity",prec_parent); break;

      case SET_CONSTRAINT_NODE:
         set_print(node,"set","constraint",prec_parent); break;

      case SET_CONSTRAINT_NAME_NODE:
         set_print(node,"set","constraint ",prec_parent); 
         strcat(pos,get_constraint((int)(node[node->left].op1.real))->name);
         pos += strlen(pos); break;

      case SET_NAMED_QUANTITY_NODE:
         set_print(node,"set","quantity",prec_parent); break;

      case UNSET_NAMED_QUANTITY_NODE:
         set_print(node,"unset","quantity",prec_parent); break;

      case SET_METHOD_INSTANCE_NODE:
         set_print(node,"set","method_instance",prec_parent); break;

      case UNSET_METHOD_INSTANCE_NODE:
         set_print(node,"unset","method_instance",prec_parent); break;

      case UNSET_CONSTRAINT_NODE:
         set_print(node,"unset","constraint",prec_parent); 
         break;

      case UNSET_CONSTRAINT_NAME_NODE:
       { char temp[100];
         sprintf(temp,"constraint %s",get_constraint(node->op3.connum)->name);
         set_print(node,"unset",temp,prec_parent); 
         pos += strlen(pos);
         break;
       }

      case UNSET_BOUNDARY_NODE:
         set_print(node,"unset","boundary",prec_parent); break;

      case UNSET_BOUNDARY_NAME_NODE:
       { char temp[100];
         sprintf(temp,"boundary %s",web.boundaries[node->op3.bdrynum].name);
         set_print(node,"unset",temp,prec_parent); 
         pos += strlen(pos);
         break;
       }

      case UNSET_TARGET_NODE:
         set_print(node,"unset","target",prec_parent); break;

      case UNSET_PRESSURE_NODE:
         set_print(node,"unset","pressure",prec_parent); break;


      case SET_NO_HESSIAN_NORMAL_NODE:
         set_print(node,"set","hessian_normal",prec_parent); break;

      case UNSET_NO_HESSIAN_NORMAL_NODE:
         set_print(node,"unset","hessian_normal",prec_parent); break;

      case SET_INIT_NODE: break;

      case SET_ATTRIBUTE_LOOP_NODE:
      { struct treenode *nnode;

         sprintf(pos,"set ");
         pos += strlen(pos);
         nnode = node + node->left;
         if ( nnode->type == WHERE_NODE ) nnode += nnode->left; /* get NEXT_ */
         exprint_recur(nnode,prec_parent);

         if ( node->right )  exprint_recur(node+node->right,prec_parent);
         if ( node[node->left].type == WHERE_NODE )
           { node+= node->left;
             sprintf(pos," where "); pos += strlen(pos);
             exprint_recur(node+node->right,prec_parent);
           }
       }
       return;

      case FOREACH_NODE:
         sprintf(pos,"foreach ");
         pos += strlen(pos);
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos," do "); 
         pos += strlen(pos);
         bracket_depth++;
         newline();
         exprint_recur(node+node->right,prec_parent);
         bracket_depth--;
         return;

      case MAX_NODE: 
         sprintf(pos,"max(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,")"); pos++;
         return;

      case MIN_NODE: 
         sprintf(pos,"min(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,")"); pos++;
         return;

      case SUM_NODE: 
         sprintf(pos,"sum(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,")"); pos++;
         return;

      case AVG_NODE: 
         sprintf(pos,"avg(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,")"); pos++;
         return;

      case COUNT_NODE:
         sprintf(pos,"count(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,")"); pos++;
         return;

      case HISTOGRAM_NODE: 
         sprintf(pos,"histogram(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,")"); pos++;
         return;

      case LOGHISTOGRAM_NODE:
         sprintf(pos,"loghistogram(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,","); pos++;
         exprint_recur(node+node->right,PREC_ARG);
         sprintf(pos,")"); pos++;
         return;

      case WHERE_NODE:
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos," where "); pos += strlen(pos);
         exprint_recur(node+node->right,prec_parent);
         return;

      case NEXT_VERTEX_NODE: case NEXT_EDGE_VERTEX_NODE: case NEXT_FACET_VERTEX_NODE:
      case NEXT_BODY_VERTEX_NODE: case NEXT_FACETEDGE_NODE:
      case NEXT_EDGE_NODE: case NEXT_VERTEX_EDGE_NODE: case NEXT_FACET_EDGE_NODE:
      case NEXT_FACET_NODE: case NEXT_VERTEX_FACET_NODE: case NEXT_EDGE_FACET_NODE:
      case NEXT_BODY_NODE: case NEXT_VERTEX_BODY_NODE: case NEXT_EDGE_BODY_NODE:
      case NEXT_FACET_BODY_NODE:
      case NEXT_BODY_FACET_NODE:
      case NEXT_BODY_EDGE_NODE:
         exprint_recur(node+node->left,prec_parent);
         if ( strcmp(node->op5.string,default_name) != 0 )
           { sprintf(pos,"%s ",node->op5.string);
             pos += strlen(pos);
           } 
         return; 
      
      case INIT_FACETEDGE_NODE: 
         sprintf(pos,"facetedges "); pos += strlen(pos);
         return;
      
      case INIT_VERTEX_NODE: 
         sprintf(pos,"vertices "); pos += strlen(pos);
         return;
      
      case INIT_EDGE_VERTEX_NODE: case INIT_FACET_VERTEX_NODE: case INIT_BODY_VERTEX_NODE:
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,".vertices "); 
         pos += strlen(pos);
         return;
      
      case INIT_EDGE_NODE: 
         sprintf(pos,"edges "); pos += strlen(pos);
         return;
      
      case INIT_VERTEX_EDGE_NODE: case INIT_FACET_EDGE_NODE: case INIT_BODY_EDGE_NODE:
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,".edges "); 
         pos += strlen(pos);
         return;
      
      case INIT_FACET_NODE: 
         sprintf(pos,"facets "); pos += strlen(pos);
         return;
      
      case INIT_VERTEX_FACET_NODE: case INIT_EDGE_FACET_NODE: case INIT_BODY_FACET_NODE:
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,".facets "); 
         pos += strlen(pos);
         return;
      
      case INIT_BODY_NODE: 
         sprintf(pos,"bodies "); pos += strlen(pos);
         break;

      case INIT_VERTEX_BODY_NODE: case INIT_EDGE_BODY_NODE: case INIT_FACET_BODY_NODE:
         exprint_recur(node+node->left,prec_parent);
         sprintf(pos,".bodies "); 
          pos += strlen(pos);
         return;
      
      case PUSH_NAMED_QUANTITY_NODE:
         sprintf(pos,"%s ",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSH_METHOD_INSTANCE_NODE:
         sprintf(pos,"%s ",METH_INSTANCE(node->op1.meth_id)->name);
         pos += strlen(pos);
         return;

      case PUSHCONST_NODE:
#ifdef FLOAT128
         sprintf(pos,"%1.*Qg",DPREC,node->op1.real);
#elif defined(LONGDOUBLE)
         sprintf(pos,"%1.*Lg",DPREC,node->op1.real);
#else
         sprintf(pos,"%1.15g",node->op1.real);
#endif 
         pos += strlen(pos);
         return;

      case PUSHPI_NODE:
         sprintf(pos,"pi");
         pos += strlen(pos);
         return;

      case PUSHE_NODE:
         sprintf(pos,"e");
         pos += strlen(pos);
         return;

      case PUSHG_NODE:
         sprintf(pos,"G");
         pos += strlen(pos);
         return;

      case COORD_NODE:
         if ( node->left )
         { exprint_recur(node+node->left,PREC_COND);
           sprintf(pos,"."); pos += strlen(pos);
         }
         if ( (vch == 'X') && (node->op2.coordnum+1 <= 3) )
           sprintf(msg,"%c",'x'+node->op2.coordnum);
         else
           sprintf(msg,"%c%d",vch,node->op2.coordnum+1);
         print_attr(node,msg);
         return;

      case PARAM_NODE:
         sprintf(msg,"p%d",node->op2.coordnum+1);
         print_attr(node,msg);
         return;

      case PUSHPARAM_NODE:
	     if ( (vch == 'X') && (node->op1.coordnum+1 <= web.sdim) && (web.sdim <= 3) )
           sprintf(msg,"%c",'x'+node->op1.coordnum);
         else
		   sprintf(msg,"%c%d",vch,node->op1.coordnum+1);
         print_attr(node,msg);
         return;

    case SET_MMODULUS_NODE:
          sprintf(pos,"%s.modulus %s ",METH_INSTANCE(node->op1.meth_id)->name,
              assign_symbol(node->op2.assigntype)); 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_QMODULUS_NODE:
          sprintf(pos,"%s.modulus %s ",GEN_QUANT(node->op1.quant_id)->name,
               assign_symbol(node->op2.assigntype)); 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_QTARGET_NODE:
          sprintf(pos,"%s.target %s ",GEN_QUANT(node->op1.quant_id)->name,
          assign_symbol(node->op2.assigntype)); 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_QVOLCONST_NODE:
          sprintf(pos,"%s.volconst %s ",GEN_QUANT(node->op1.quant_id)->name,
          assign_symbol(node->op2.assigntype)); 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

    case SET_QTOLERANCE_NODE:
          sprintf(pos,"%s.tolerance %s ",GEN_QUANT(node->op1.quant_id)->name,
          assign_symbol(node->op2.assigntype)); 
          pos += strlen(pos);
          exprint_recur(node+node->left,PREC_ASSIGN);
          break;

      case PUSHQFIXED_NODE:
         sprintf(pos,"%s.fixed",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQENERGY_NODE:
         sprintf(pos,"%s.energy",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQINFO_ONLY_NODE:
         sprintf(pos,"%s.info_only",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQCONSERVED_NODE:
         sprintf(pos,"%s.conserved",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQPRESSURE_NODE:
         sprintf(pos,"%s.pressure",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQTARGET_NODE:
         sprintf(pos,"%s.target",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHMVALUE_NODE:
         sprintf(pos,"%s.value",METH_INSTANCE(node->op1.meth_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQVALUE_NODE:
         sprintf(pos,"%s.value",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHMMODULUS_NODE:
         sprintf(pos,"%s.modulus",GEN_QUANT(node->op1.meth_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQMODULUS_NODE:
         sprintf(pos,"%s.modulus",GEN_QUANT(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHQVOLCONST_NODE:
         sprintf(pos,"%s.volconst",globals(node->op1.quant_id)->name);
         pos += strlen(pos);
         return;

      case PUSHDELTA_NODE:
         sprintf(pos,"%s.pdelta",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;

      case PUSH_PARAM_SCALE_NODE:
         sprintf(pos,"%s.pscale",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;

      case PUSH_PARAM_FIXED_NODE:
         sprintf(pos,"%s.fixed",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;

      case PUSH_PARAM_FORCE_NODE:
         sprintf(pos,"%s.p_force",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;

      case PUSH_PARAM_VELOCITY_NODE:
         sprintf(pos,"%s.p_velocity",globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;

      case PUSH_PARAM_EXTRA_NODE:
         sprintf(pos,"%s.%s",globals(node->op1.name_id)->name,
              EXTRAS(0)[node->op2.extranum].name);
         pos += strlen(pos);
         return;

      case ELEMENT_IDENT_NODE:
         { struct global *g = globals(node->op3.name_id); 
           sprintf(pos,"%s",g->name);
           pos += strlen(pos);
           return;
         }
      case PUSHGLOBAL_NODE:
      case STRINGGLOBAL_NODE:
       { struct global *g = globals(node->op1.name_id);
         if ( g->flags & QUANTITY_NAME )
           sprintf(pos,"total %s",g->name);
         else sprintf(pos,"%s",g->name);
         pos += strlen(pos);
         return;
       }
      case PUSH_PERM_GLOBAL_NODE:
      case PERM_STRINGGLOBAL_NODE:
         sprintf(pos,"%s",perm_globals(node->op1.name_id)->name);
         pos += strlen(pos);
         return;

      case USERFUNC_NODE:
         sprintf(pos,"usr%d",node->op1.userfunc+1);
         pos += strlen(pos);
         return;

      case DYNAMIC_LOAD_FUNC_NODE:
         sprintf(pos,"%s",globals(node->op2.name_id)->name);
         pos += strlen(pos);
         return;

      case ARRAY_ADD_NODE:
         binary_print(node,prec_parent,PREC_ADD," + ",PREC_ADD);
         return;

      case ARRAY_SUBTRACT_NODE:
         binary_print(node,prec_parent,PREC_ADD," - ",PREC_ADD);
         return;

      case ARRAY_MULTIPLY_NODE:
      case ARRAY_SCALAR_MULTIPLY_NODE:
         binary_print(node,prec_parent,PREC_MUL," * ",PREC_MUL);
         return;

      case ARRAY_SCALAR_DIVIDE_NODE: // special, since denominator got swapped to first
        if ( prec_parent > PREC_DIV ) 
        { sprintf(pos,"(");
          pos += strlen(pos);
        }
        exprint_recur(node+node->right,PREC_DIV);
        sprintf(pos,"/");
        pos += strlen(pos);
        exprint_recur(node+node->left,PREC_DIV+1);
        if ( prec_parent > PREC_DIV )
        { sprintf(pos,")");
          pos += strlen(pos);
        }
        break;

      case PLUS_NODE:
         binary_print(node,prec_parent,PREC_ADD," + ",PREC_ADD);
         return;

      case MINUS_NODE:
         binary_print(node,prec_parent,PREC_SUB," - ",PREC_SUB+1);
         return;

      case EQUATE_NODE:
         binary_print(node,prec_parent,PREC_ASSIGN," = ",PREC_ASSIGN+1);
         return;

      case TIMES_NODE:
         binary_print(node,prec_parent,PREC_MUL,"*",PREC_MUL);
         return;

      case DIVIDE_NODE:
         binary_print(node,prec_parent,PREC_DIV,"/",PREC_DIV+1);
         return;

      case REALMOD_NODE: 
         binary_print(node,prec_parent,PREC_MOD,"%",PREC_MOD+1);
         return;

      case IMOD_NODE: 
         binary_print(node,prec_parent,PREC_MOD," imod ",PREC_MOD+1);
         return;

      case IDIV_NODE:
         binary_print(node,prec_parent,PREC_DIV," idiv ",PREC_DIV+1);
         return;

      case COND_ELSE_NODE:
         binary_print(node,prec_parent,PREC_COND,"):(",PREC_COND);
         strcat(pos++,")");
         return;

      case COND_EXPR_NODE:
         strcat(pos++,"(");
         binary_print(node,prec_parent,PREC_COND,")?(",PREC_COND);
         return;

      case COND_TEST_NODE:
         exprint_recur(node+node->left,PREC_COND);
         return;

      case INV_NODE:
         exprint_recur(node+node->left,PREC_POW);
         sprintf(pos,"^(-1)");
         pos += strlen(pos);
         return;

      case INTPOW_NODE:
         exprint_recur(node+node->left,PREC_POW);
         sprintf(pos,"^%1d",node->op1.intpow);
         pos += strlen(pos);
         return;

      case POW_NODE:
         binary_print(node,prec_parent,PREC_POW,"**",PREC_POW);
         return;

      case MAXIMUM_NODE:
         strcat(pos,"maximum(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,",");
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,")");
         return;

      case MINIMUM_NODE:
         strcat(pos,"minimum(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,",");
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,")");
         return;

      case INCOMPLETE_ELLIPTICF_NODE:
         sprintf(pos,"incompleteEllipticF(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,",");
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,")");
         return;

      case INCOMPLETE_ELLIPTICE_NODE:
         sprintf(pos,"incompleteEllipticE(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,",");
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,")");
         return;

      case ATAN2_NODE:
         sprintf(pos,"atan2(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,",");
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,")");
         return;

      case WRAP_COMPOSE_NODE:
         sprintf(pos,"wrap_compose(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos++,",");
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,")");
         return;

      case WRAP_INVERSE_NODE:
         sprintf(pos,"wrap_inverse(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case SQR_NODE:
         sprintf(pos,"(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")^2");
         pos += strlen(pos);
         return;

      case SQRT_NODE:
         sprintf(pos,"sqrt(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case FLOOR_NODE:
         sprintf(pos,"floor(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case CEIL_NODE:
         sprintf(pos,"ceil(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case ABS_NODE:
         sprintf(pos,"abs(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case SINH_NODE:
         sprintf(pos,"sinh(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case COSH_NODE:
         sprintf(pos,"cosh(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case TANH_NODE:
         sprintf(pos,"tanh(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case ACOSH_NODE:
         sprintf(pos,"acosh(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case ASINH_NODE:
         sprintf(pos,"asinh(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case ATANH_NODE:
         sprintf(pos,"atanh(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case SIN_NODE:
         sprintf(pos,"sin(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case COS_NODE:
         sprintf(pos,"cos(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case TAN_NODE:
         sprintf(pos,"tan(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case EXP_NODE:
         sprintf(pos,"exp(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case LOG_NODE:
         sprintf(pos,"log(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case ASIN_NODE:
         sprintf(pos,"asin(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case ACOS_NODE:
         sprintf(pos,"acos(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;

      case ATAN_NODE:
         sprintf(pos,"atan(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;
      
      case ELLIPTICK_NODE:
         sprintf(pos,"ellipticK(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;
      
      case ELLIPTICE_NODE:
         sprintf(pos,"ellipticE(");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         sprintf(pos,")");
         pos += strlen(pos);
         return;
      
      case CHS_NODE:
         sprintf(pos," -");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_UMINUS);
         return;
      
      case NOT_NODE:
         sprintf(pos," not ");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_NOT);
         return;
      
      case VIEW_MATRIX_NODE:
         sprintf(pos,"view_matrix[");
         pos += strlen(pos);
         exprint_recur(node+node->left,PREC_ARG);
         strcat(pos,"]["); pos += 2;
         exprint_recur(node+node->right,PREC_ARG);
         strcat(pos++,"]");
         return;

      case GET_EXTRA_ATTR_NODE:
         ex = EXTRAS(node->op2.eltype) + node->op3.extranum;
         print_attr(node,ex->name);
         if ( node->left ) 
           exprint_recur(node+node->left,prec_parent);       
         break;

      case ON_QUANTITY_NODE:
         print_attr(node,"on_quantity ");
         strcat(pos,GEN_QUANT(node->op2.quant_id)->name);
         pos += strlen(pos);
         return;

      case ON_METHOD_INSTANCE_NODE:
         print_attr(node,"on_method_instance ");
         strcat(pos,METH_INSTANCE(node->op2.meth_id)->name);
         pos += strlen(pos);
         return;

      case ON_CONSTRAINT_NODE:
         print_attr(node,"on_constraint ");
         exprint_recur(node+node->left,PREC_INDEX);
         return;

      case ON_CONSTRAINT_NAME_NODE:
         print_attr(node,"on_constraint ");
         strcat(pos,get_constraint(node->op3.connum)->name);
         pos += strlen(pos);
         return;

      case HIT_CONSTRAINT_NODE:
         print_attr(node,"hit_constraint ");
         exprint_recur(node+node->left,PREC_INDEX);
         return;

      case HIT_CONSTRAINT_NAME_NODE:
         print_attr(node,"hit_constraint ");
         strcat(pos,get_constraint(node->op3.connum)->name);
         pos += strlen(pos);
         return;

      case CONSTRAINT_VALUE_NODE:
         print_attr(node,"value_of_constraint ");
         exprint_recur(node+node->left,PREC_INDEX);
         return;

      case CONSTRAINT_NAME_VALUE_NODE:
         print_attr(node,"constraint ");
         strcat(pos,get_constraint(node->op3.connum)->name);
         strcat(pos," value");
         pos += strlen(pos);
         return;
      

      case ON_BOUNDARY_NODE:
         print_attr(node,"on_boundary ");
         exprint_recur(node+node->left,PREC_INDEX);
         return;

      case ON_BOUNDARY_NAME_NODE:
         print_attr(node,"on_boundary ");
         strcat(pos,web.boundaries[node->op3.bdrynum].name);
         pos += strlen(pos);
         return;

      case QUALIFIED_ATTRIBUTE_NODE:
         exprint_recur(node+node->left,prec_parent);
         strcat(pos,"."); pos++;
         exprint_recur(node+node->right,prec_parent);
         return;

      case GET_MIDV_NODE:
         print_attr(node,"midv");
         return;

      case GET_TRIPLE_PT_NODE:
         print_attr(node,"triple_point");
         return;

      case GET_TETRA_PT_NODE:
         print_attr(node,"tetra_point");
         return;

      case GET_AXIAL_POINT_NODE:
         print_attr(node,"axial_point");
         return;

      case GET_FIXED_NODE:
         print_attr(node,"fixed");
         return;

      case GET_BARE_NODE:
         print_attr(node,"bare");
         return;

      case GET_NO_DISPLAY_NODE:
         print_attr(node,"no_display");
         return;

      case GET_NONCONTENT_NODE:
         print_attr(node,"noncontent");
         return;

      case GET_HIT_PARTNER_NODE:
         print_attr(node,"hit_partner");
         return;

      case GET_NO_REFINE_NODE:
         print_attr(node,"no_refine");
         return;

      case GET_NO_TRANSFORM_NODE:
         print_attr(node,"no_transform");
         return;

      case GET_OPACITY_NODE:
         print_attr(node,"opacity");
         return;

      case GET_ORIGINAL_NODE:
         print_attr(node,"original");
         return;


      case GET_ID_NODE:
         print_attr(node,"id");
         return;

      case GET_OID_NODE:
         print_attr(node,"oid");
         return;

      case GET_VALENCE_NODE:
         print_attr(node,"valence");
         return;

      case GET_COLOR_NODE:
         print_attr(node,"color");
         return;

      case GET_FRONTCOLOR_NODE:
         print_attr(node,"frontcolor");
         return;

      case GET_BACKCOLOR_NODE:
         print_attr(node,"backcolor");
         return;

      case GET_FRONTBODY_NODE:
         print_attr(node,"frontbody");
         return;

      case GET_BACKBODY_NODE:
         print_attr(node,"backbody");
         return;

      case GET_ORIENTATION_NODE:
         print_attr(node,"orientation");
         return;

      case GET_SHOW_NODE:
         print_attr(node,"show");
         return;

      case GET_LENGTH_NODE:
         print_attr(node,"length");
         return;

      case GET_MEANCURV_NODE:
         print_attr(node,"mean_curvature");
         return;

      case GET_FIXEDVOL_NODE:
         print_attr(node,"volfixed");
         return;

      case GET_NO_HESSIAN_NORMAL_NODE:
         print_attr(node,"no_hessian_normal");
         return;

      case GET_MID_EDGE_NODE:
         print_attr(node,"mid_edge");
         return;

      case GET_MID_FACET_NODE:
         print_attr(node,"mid_facet");
         return;

      case GET_WRAP_NODE:
         print_attr(node,"wrap");
         return;

      case GET_SQ_MEAN_CURV_NODE:
         print_attr(node,"sqcurve");
         return;

      case GET_DIHEDRAL_NODE:
         print_attr(node,"dihedral");
         return;

      case GET_AREA_NODE:
         print_attr(node,"area");
         return;

      case GET_VOLUME_NODE:
         print_attr(node,"volume");
         return;

      case GET_VOLCONST_NODE:
         print_attr(node,"volconst");
         return;

      case GET_TARGET_NODE:
         print_attr(node,"target");
         return;

      case GET_MPI_TASK_NODE:
         print_attr(node,"mpi_task");
         return;

      case GET_PRESSURE_NODE:
         print_attr(node,"pressure");
         return;

      case GET_DENSITY_NODE:
         print_attr(node,"density");
         return;

      case GET_PHASE_NODE:
         print_attr(node,"phase");
         return;

      case GET_QUANTITY_NODE:
         print_attr(node,GEN_QUANT(node->op2.quant_id)->name);
         return;

      case GET_INSTANCE_NODE:
         print_attr(node,METH_INSTANCE(node->op2.meth_id)->name);
         return;

      case INDEXSET_NODE:
      case DIMENSIONSET_NODE:
         if ( node->right )
         { exprint_recur(node+node->left,prec_parent);
           strcat(pos++,"[");
           exprint_recur(node+node->right,PREC_ARG);
           strcat(pos++,"]");
         } else
         {
           strcat(pos++,"[");
           exprint_recur(node+node->left,PREC_ARG);
           strcat(pos++,"]");
         }
         break;

      case UNSET_FRONTBODY_NODE:
         set_print(node,"unset","frontbody",prec_parent); break;
 
      case UNSET_BACKBODY_NODE:
         set_print(node,"unset","backbody",prec_parent); break;
 
      case SET_FIXED_NODE: set_print(node,"set","fixed",prec_parent); break;
      case UNSET_FIXED_NODE: set_print(node,"unset","fixed",prec_parent); break;
  
      case SET_ATTRIBUTE_NODE:
         strcat(pos,"set "); pos += 4;
      case SET_ATTRIBUTE_A_NODE: /* single element assign */
      case SET_ATTRIBUTE_L_NODE:  /* skip printing set when in set loop */
        if ( pos[-1] != '.' ) strcat(pos++," "); /* just to be sure */
        switch ( node->op2.attr_kind )
        { case SET_DENSITY_NODE: print_set_attr(node,"density"); break;
          case SET_EXTRA_ATTR_NODE: 
            { ex = EXTRAS(node->op3.extra_info>>ESHIFT)
                        +(node->op3.extra_info&0xFF);
              print_attr(node,ex->name);
              if ( node->right )        
                exprint_recur(node+node->right,prec_parent);
            }
            break;
          case SET_PHASE_NODE : print_set_attr(node,"phase"); break;
          case SET_WRAP_NODE : print_set_attr(node,"wrap"); break;
          case SET_ORIENTATION_NODE : print_set_attr(node,"orientation"); break;
          case SET_TARGET_NODE: print_set_attr(node,"target"); break;
          case SET_VOLCONST_NODE: print_set_attr(node,"volconst"); break;
          case SET_PRESSURE_NODE: print_set_attr(node,"pressure"); break;
          case SET_OPACITY_NODE: print_set_attr(node,"opacity"); break;
          case SET_COLOR_NODE: print_set_attr(node,"color"); break;
          case SET_ORIGINAL_NODE: print_set_attr(node,"original"); break;
          case SET_FRONTBODY_NODE: print_set_attr(node,"frontbody"); break;
          case SET_BACKBODY_NODE: print_set_attr(node,"backbody"); break;
          case SET_FRONTCOLOR_NODE: print_attr(node,"frontcolor"); break;
          case SET_BACKCOLOR_NODE: print_set_attr(node,"backcolor"); break; 
          case SET_CONSTRAINT_NODE: 
             { struct constraint *con;
               print_set_attr(node,"constraint "); 
               if ( node[node->left].type == PUSHCONST_NODE )
               { int cnum = (int)(node[node->left].op1.real);
                 con = get_constraint(cnum);
                 if ( (cnum <= web.maxcon) && (con->attr & NAMED_THING) )
                 { strcat(pos,con->name);
                   pos += strlen(pos); 
                   return;
                 } /* else recursion takes care of expression number constant */
               }
               pos += strlen(pos); 
               break;
             }
          case SET_BOUNDARY_NODE: 
             { struct boundary *bdry;
               print_set_attr(node,"boundary "); 
               if ( node[node->left].type == PUSHCONST_NODE )
               { int bnum = (int)(node[node->left].op1.real);
                 bdry = web.boundaries+bnum;
                 if ( (bnum <= web.bdrymax) && (bdry->attr & NAMED_THING) )
                 { strcat(pos,bdry->name);
                   pos += strlen(pos); 
                   return;
                 } /* else recursion takes care of expression number constant */
               }
               pos += strlen(pos); 
               break;
             }
          case SET_FIXED_NODE: print_set_attr(node,"fixed"); break;
          case SET_NO_HESSIAN_NORMAL_NODE: print_set_attr(node,"no_hessian_normal"); break;
          case SET_BARE_NODE: print_set_attr(node,"bare"); break;
          case SET_NO_REFINE_NODE: print_set_attr(node,"no_refine"); break;
          case SET_NO_TRANSFORM_NODE: print_set_attr(node,"no_transform"); break;
          case SET_HIT_PARTNER_NODE: print_set_attr(node,"hit_partner"); break;
          case SET_NONCONTENT_NODE: print_set_attr(node,"noncontent"); break;
          case SET_NO_DISPLAY_NODE: print_set_attr(node,"no_display"); break;
          case SET_TETRA_PT_NODE: print_set_attr(node,"tetra_point"); break;
          case SET_AXIAL_POINT_NODE: print_set_attr(node,"axial_point"); break;
          case SET_TRIPLE_PT_NODE: print_set_attr(node,"triple_point"); break;
          case SET_COORD_1_NODE: 
             if ( node->right )
             { print_set_attr(node,"x"); 
               exprint_recur(node+node->right,PREC_INDEX);
               break;
             }
             else print_set_attr(node,"x"); 
             break;
          case SET_COORD_2_NODE: print_set_attr(node,"y"); break;
          case SET_COORD_3_NODE: if ( web.sdim <= 2 ) print_set_attr(node,"x3");
               else print_set_attr(node,"z"); break;
          case SET_COORD_4_NODE: print_set_attr(node,"x4"); break;
          case SET_COORD_5_NODE: print_set_attr(node,"x5"); break;
          case SET_COORD_6_NODE: print_set_attr(node,"x6"); break;
          case SET_COORD_7_NODE: print_set_attr(node,"x7"); break;
          case SET_COORD_8_NODE: print_set_attr(node,"x8"); break;
          case SET_PARAM_1_NODE: 
             if ( node->right )
             { print_set_attr(node,"p1["); pos += 3;
               exprint_recur(node+node->right,PREC_ARG);
               strcat(pos,"]"); pos += strlen(pos);
               break;
             }
             else print_set_attr(node,"p1"); 
             break;
          case SET_PARAM_2_NODE: print_set_attr(node,"p2"); break;
          case SET_PARAM_3_NODE: print_set_attr(node,"p3"); break;
          case SET_PARAM_4_NODE: print_set_attr(node,"p4"); break;
          default:  
              sprintf(errmsg,"Internal error: bad SET_ATTRIBUTE type %d.\n",
                node->op2.attr_kind);
              kb_error(1655,errmsg,RECOVERABLE);

        }
        if ( node->type == SET_ATTRIBUTE_A_NODE )
          { switch ( node[1].op1.assigntype )
             { case ASSIGN_OP: strcat(pos," := "); break;
               case PLUSASSIGN_OP: strcat(pos," += "); break;
               case SUBASSIGN_OP: strcat(pos," -= "); break;
               case MULTASSIGN_OP: strcat(pos," *= "); break;
               case DIVASSIGN_OP: strcat(pos," /= "); break;
             }
             pos += 4; 
             strcat(pos," ("); pos += 2;  // kludge for "-" after []
             exprint_recur(node+node->left,PREC_ASSIGN);
             strcat(pos++,")");
          }
        else
        {
          strcat(pos," ("); pos += 2;
          if ( node->left )  exprint_recur(node+node->left,PREC_ARG);
          strcat(pos++,")");
        }
        return;

     case SINGLE_ASSIGN_NODE: 
         exprint_recur(node+node->left,prec_parent);
         *(pos++) = '.'; *pos = 0;
         exprint_recur(node+node->right,prec_parent);
         return;

     case EQ_NODE:
         binary_print(node,prec_parent,PREC_COMP," == ",PREC_COMP);
         return;
     case NE_NODE:
         binary_print(node,prec_parent,PREC_COMP," != ",PREC_COMP);
         return;
     case GE_NODE:
         binary_print(node,prec_parent,PREC_COMP," >= ",PREC_COMP);
         return;
     case LE_NODE:
         binary_print(node,prec_parent,PREC_COMP," <= ",PREC_COMP);
         return;
     case GT_NODE:
         binary_print(node,prec_parent,PREC_COMP," > ",PREC_COMP);
         return;
     case LT_NODE:
         binary_print(node,prec_parent,PREC_COMP," < ",PREC_COMP);
         return;
     case AND_NODE:
         binary_print(node,prec_parent,PREC_AND," && ",PREC_AND);
         return;
     case OR_NODE:
         binary_print(node,prec_parent,PREC_OR," || ",PREC_OR);
         return;
     case CONJUNCTION_END_NODE:     
         exprint_recur(node+node->left,prec_parent);
         return;

     default: 
         sprintf(pos,"(unknown)");
         pos += strlen(pos);
         sprintf(errmsg,"Printing of expression node type %s (%d) unimplemented.\n",
         tokname(node->type),node->type);
         kb_error(1656,errmsg,WARNING);

         return;
     }
  return ; /* shouldn't happen */
} /* end exprint_recur */

/**************************************************************************
*
* function: binary_print()
*
* purpose: print binary operation with parentheses if needed.
*
*/

void binary_print(
  struct treenode *node,
  int prec_parent,
  int prec1,
  char *op,
  int prec2
)
{
  if ( prec_parent > prec1 ) 
   { sprintf(pos,"(");
     pos += strlen(pos);
   }
  exprint_recur(node+node->left,prec1);
  sprintf(pos,"%s",op);
  pos += strlen(pos);
  exprint_recur(node+node->right,prec2);
  if ( prec_parent > prec1 )
    { sprintf(pos,")");
      pos += strlen(pos);
    }
  return;
} // end binary_print()

/**************************************************************************
*
* function: set_print()
*
* purpose: print SET type command, with possible WHERE clause.
*
*/

void set_print(
  struct treenode *node,
  char *keyw, char *attrw,
  int prec_parent
)
{ struct treenode *nnode;
  sprintf(pos,"%s ",keyw);
  pos += strlen(pos);
  nnode = node + node->left;
  if ( nnode->type == WHERE_NODE ) nnode += nnode->left; /* get NEXT_ */
  exprint_recur(nnode,prec_parent);
  sprintf(pos," %s ",attrw); 
  pos += strlen(pos);
  if ( node->right )  exprint_recur(node+node->right,prec_parent);
  if ( node[node->left].type == WHERE_NODE )
  { node+= node->left;
    sprintf(pos," where "); pos += strlen(pos);
    exprint_recur(node+node->right,prec_parent);
  }
  return;
} // end set_print()

/**************************************************************************
*
* function: print_attr()
*
* purpose: add attribute name to output
*
*/
void print_attr(
  struct treenode *node,
  char *word
)
{
  sprintf(pos,"%s",word); 
  pos += strlen(pos); return;
} // end print_attr()

/**************************************************************************
*
* function: print_set_attr()
*
* purpose: add attribute name to output
*
*/
void print_set_attr(node,word)
struct treenode *node;
char *word;
{
  sprintf(pos,"%s",word); 
  pos += strlen(pos); return;
} // end print_set_attr()

