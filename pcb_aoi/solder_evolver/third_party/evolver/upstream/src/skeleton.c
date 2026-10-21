/*************************************************************
*  This file is part of the Surface Evolver source code.     *
*  Programmer:  Ken Brakke, brakke@susqu.edu                 *
*************************************************************/

/*****************************************************************
*
*  File: skeleton.c
*
*  Purpose: Variables and functions for skeleton element handling
*/

#include "include.h"

/* non-inlined versions of inlinable functions */
#ifndef INLINE
#define INLINE
#include "inline.h"
#endif

/* extra attribute type sizes, matching order in skeleton.h */

int datatype_size[NUMDATATYPES] = 
  {0,sizeof(REAL),sizeof(int),sizeof(unsigned long), sizeof(unsigned char),
   sizeof(unsigned short), sizeof(unsigned int),
   sizeof(long int),sizeof(char),sizeof(short),0,sizeof(char*),sizeof(char*),
   sizeof(vertex_id),sizeof(edge_id),sizeof(facet_id),sizeof(body_id),
   sizeof(facetedge_id),sizeof(element_id),sizeof(int),sizeof(int),sizeof(int),
   sizeof(int), sizeof(int), sizeof(int)
  };

char *datatype_name[NUMDATATYPES] =
  { " ","real","integer","ulong_int","uchar","ushort_int","uint",
    "long_int","char","short_int"," ","string","pointer", "vertex_type",
    "edge_type","facet_type","body_type","facetedge_type", "element_id",
    "boundary_type","constraint_type","quantity_type","method_instance_type",
    "procedure_type" , "array"
  };

/* quadratic interpolation coefficients */
/* partials of interpolation polynomials at midpoints of patch edges */
/* control points are numbered 0 to 5 counterclockwise */
/* midpoints are numbered 0 to 2 counterclockwise after control pt 0 */
/* dip[midpoint][control point][partial]  */

REAL dip[FACET_EDGES][FACET_CTRL][2] = {
    { { -0.5, -0.5 }, { 0.0, -1.0 }, { 0.5, 0.0 }, { 0.0, 1.0 },
      { 0.0, -0.5 }, { 0.0, 1.0 } },
    { { 0.5, 0.5 }, { -1.0, -1.0 }, { 0.5, 0.0  }, { 1.0, 1.0 },
      { 0.0, 0.5 }, { -1.0, -1.0 } },
    { { -0.5, -0.5 }, { 1.0, 0.0 }, { -0.5, 0.0 }, { 1.0, 0.0 },
      { 0.0, 0.5 }, { -1.0, 0.0 } }
  };

/****************************************************************
* function: Ord()
* purpose: return ordinal of element; useful in debugging.
*/
int Ord(element_id id)
{ return loc_ordinal(id); }

/******************************************************************
* function: vloop()
* purpose:  prints vertex edge loop; useful in debugger 
*/
void vloop(element_id v_id)
{ edge_id e_id,ee_id;
  int n = 0;
  e_id = get_vertex_edge(v_id);
  if ( !valid_id(e_id) ) 
     { puts("No valid edge on vertex."); return; }
  ee_id = e_id;
  do { printf("%d ",inverted(ee_id)?-(Ord(ee_id)+1): (Ord(ee_id)+1)); 
       ee_id = get_next_tail_edge(ee_id);
       if ( ++n > web.skel[EDGE].count )
       { puts("Unclosed loop."); break;}
     } while ( ee_id != e_id );
  printf("\n");
} // end vloop()

/******************************************************************
* function: set_facet_body()
* purpose: Set the front body of a facet; null body also possible.
*          Takes care of body facet list, and body quantity if 
*          everything_quantities.
*/
void set_facet_body(
  facet_id f_id,
  body_id  b_id  /* may be invalid for unsetting */
)
{ body_id bb_id;  // the old body
  
  if ( web.skel[BODY].count == 0 ) return;
  if ( !valid_id(f_id) ) return;

#ifdef MPI_EVOLVER
  if ( id_task(f_id) != this_task )
	  return; /* imported facets not part of local body chains */
#endif

  bb_id = get_facet_body(f_id);
  if ( equal_id(bb_id,b_id) ) return;

  /* check if old body of this facet links to this facet */
  if ( valid_id(bb_id) )
  { 
    facet_id ff_id = get_body_facet(bb_id);
    facet_id next_f,prev_f;
    
    if ( valid_id(ff_id) ) 
    {
      if ( equal_id(f_id,ff_id) )  /* need to give body new link */
      { 
        prev_f = get_prev_body_facet(f_id);
        if ( equal_id(prev_f,f_id) || !valid_id(prev_f) ||
         !equal_id(bb_id,get_facet_body(prev_f))  )
            set_body_facet(bb_id,NULLID);
        else
          set_body_facet(bb_id,prev_f);
      }
     
      /* remove from old body facet list */
      next_f = get_next_body_facet(f_id);
      prev_f = get_prev_body_facet(f_id);
      set_next_body_facet(prev_f,next_f);
      set_prev_body_facet(next_f,prev_f);
    }
  }
 
  /* insert in new body list */
  if ( valid_id(b_id) )
  { 
    facet_id nextf = get_body_facet(b_id);
    if ( valid_id(nextf) )
    { facet_id prevf = get_prev_body_facet(nextf);
      set_next_body_facet(f_id,nextf);
      set_prev_body_facet(nextf,f_id);
      set_next_body_facet(prevf,f_id);
      set_prev_body_facet(f_id,prevf);
    }
    else /* first facet for body */
    { 
      set_next_body_facet(f_id,f_id);
      set_prev_body_facet(f_id,f_id);
      set_body_facet(b_id,f_id);
    }
  }

  if ( inverted(f_id) )  
       F_ELID(f_id,F_BODY_LIST_ATTR)[1] = b_id;
  else  
       F_ELID(f_id,F_BODY_LIST_ATTR)[0] = b_id;
 
  if ( everything_quantities_flag )
  {
     if ( web.representation == SOAPFILM )
     { facetedge_id fe = get_facet_fe(f_id);
       edge_id e_id;
       int i;
       unsigned int k;
       conmap_t *map;
       char name[100];

       if ( inverted(f_id) )  /* back facet */
       { 
          if ( valid_id(bb_id) ) /* cancel out if already there */
            apply_method_num(inverse_id(f_id),get_body_volmeth(bb_id)); 
          if ( valid_id(b_id) )
            apply_method_num(f_id,get_body_volmeth(b_id)); 
       }
       else /* front facet */
       { 
          if ( valid_id(bb_id) )
            apply_method_num(inverse_id(f_id),get_body_volmeth(bb_id)); 
          if ( valid_id(b_id) )
            apply_method_num(f_id,get_body_volmeth(b_id)); 
       }

       /* and content integrands on edges */
       for ( i = 0 ; i < FACET_EDGES ; i++, fe = get_next_edge(fe))
       { e_id = get_fe_edge(fe);
         if ( !get_eattr(e_id) & BDRY_CONTENT ) continue;
         map = get_e_constraint_map(e_id);
         for ( k = 1 ; k <= map[0] ; k++ )
         { struct constraint *con = get_constraint(map[k]);
           int inst_num;
           if ( con->attr & CON_CONTENT )
           { if ( valid_id(bb_id) )
             { if ( con->attr & NAMED_THING )
                 sprintf(name,"body_%d_%s_meth",ordinal(bb_id)+1,con->name);
               else
                 sprintf(name,"body_%d_con_%d_meth",ordinal(bb_id)+1,map[k]&CONMASK);
               inst_num = find_method_instance(name);
               if ( inst_num < 0 )
                  inst_num = create_body_constraint_content_method(bb_id,map[k]);
 attach_method(get_body_volquant(bb_id),name);
               apply_method_num(inverse_id(e_id),inst_num); /* cancel */
             }
             if ( valid_id(b_id) )
             { if ( con->attr & NAMED_THING ) 
                 sprintf(name,"body_%d_%s_meth",ordinal(b_id)+1,con->name);
               else 
                 sprintf(name,"body_%d_con_%d_meth",ordinal(b_id)+1,map[k]&CONMASK);
               inst_num = find_method_instance(name);
               if ( inst_num < 0 )
                  inst_num = create_body_constraint_content_method(b_id,map[k]);
 attach_method(get_body_volquant(b_id),name);
               apply_method_num(e_id,inst_num);
             }
           }
         }
       }
     }
     else if ( web.representation == STRING )
     {
       vertex_id first_v = NULLID, last_v = NULLID;
       
       // Apply body method to edges of face
       facetedge_id first_fe = get_facet_fe(f_id);
       if ( valid_id(first_fe) )
       { edge_id e_id;
         if ( inverted(f_id) )
         { facetedge_id nextfe=first_fe;
           /* go backwards */
           first_v = get_fe_headv(first_fe);
           do
           { e_id = get_fe_edge(nextfe);
             check_edge_vol_methods(e_id);
             last_v = get_fe_tailv(nextfe);
             nextfe = get_prev_edge(nextfe);
           } while ( valid_id(nextfe) && !equal_id(nextfe,first_fe) );
           if ( valid_id(nextfe) )
           { first_v = last_v = NULLID;
           }
         }
         else /* forward */
         { facetedge_id nextfe=first_fe;
           /* go forwards */
           first_v = get_fe_tailv(first_fe);
           do
           { e_id = get_fe_edge(nextfe);
             check_edge_vol_methods(e_id);
             last_v = get_fe_headv(nextfe);
             nextfe = get_next_edge(nextfe);
           } while ( valid_id(nextfe) && !equal_id(nextfe,first_fe) );
           if ( valid_id(nextfe) )
           { first_v = last_v = NULLID;
           }
         }

         // Content integrands on endpoints
         if ( valid_id(first_v) )
           assure_v_constraints(first_v);
         if ( valid_id(last_v) )
           assure_v_constraints(last_v);
        
       }      
     }
  }
   
  top_timestamp = ++global_timestamp;

}  // end set_facet_body()

/**************************************************************************
*
* Function: set_facet_fe()
*
* Purpose: Set facet link to a facetedge.
*
*/
void set_facet_fe(
  facet_id f_id,
  facetedge_id fe
)
{
  if ( inverted(f_id) ) { invert(fe); invert(f_id); }
  fptr(f_id)->fe_id = fe;
  if ( web.representation == STRING )
  { body_id b_id = get_facet_body(f_id);
    if ( valid_id(b_id) )
       set_body_facet(b_id,f_id);
    b_id = get_facet_body(inverse_id(f_id));
    if ( valid_id(b_id) )
       set_body_facet(b_id,inverse_id(f_id));
  }
  top_timestamp = ++global_timestamp;
} // end set_facet_fe()

/**************************************************************************
*
* Function: get_vertex_length_star()
*
* Purpose: Get total lengths of edges around vertex.  For string model
*          in area normalization mode.
*/
REAL get_vertex_length_star(vertex_id v_id)
{ edge_id e_id = get_vertex_edge(v_id);
  edge_id firste = e_id;
  REAL star = 0.0;
  if ( get_vattr(v_id) & (Q_MIDPOINT|Q_MIDEDGE) ) return get_edge_length(e_id);
  if ( !valid_id(e_id) ) return 0.0;
  do { star += get_edge_length(e_id); e_id = get_next_tail_edge(e_id);}
  while ( !equal_element(e_id,firste) );
  return star;
} // end get_vertex_length_star()

/**************************************************************************
*
* Function: get_vertex_area_star()
*
* Purpose: get total facet area around vertex, for area normalization mode.
*/
REAL get_vertex_area_star(vertex_id v_id)
{ 
  REAL star = 0.0;

  if ( web.representation == STRING ) 
      return get_vertex_length_star(v_id);
  else if ( web.representation == SOAPFILM )
  {
     facetedge_id fe_id = get_vertex_first_facet(v_id);
     facetedge_id firstfe = fe_id;
     facet_id f_id;
     if ( !valid_id(fe_id) ) return 0.0;
     do 
     { f_id = get_fe_facet(fe_id);
       star += get_facet_area(f_id); 
       fe_id = get_next_vertex_facet(v_id,fe_id);
     }
     while ( !equal_element(fe_id,firstfe) );
  }
  else /* SIMPLEX */
  {
     facet_id f_id = get_vertex_first_facet(v_id);
     facet_id firstf = f_id;
     if ( !valid_id(f_id) ) return 0.0;
     do 
     { star += get_facet_area(f_id); 
       f_id = get_next_vertex_facet(v_id,f_id);
     }
     while ( !equal_element(f_id,firstf) );
  } 
  return star;
} // end get_vertex_area_star()

/**************************************************************************
*
* Function: get_vertex_fvalence()
*
* Purpose: get number of facets around a vertex
*/
int get_vertex_fvalence(vertex_id v_id)
{ 
  int valence = 0;
  facet_id f_id = get_vertex_first_facet(v_id);
  facet_id firstf = f_id;

  if ( get_vattr(v_id) & Q_MIDFACET ) return 1;

  if ( !valid_id(f_id) ) return 0;
  do 
  { valence++; f_id = get_next_vertex_facet(v_id,f_id);}
  while ( !equal_element(f_id,firstf) );
  return valence; 
} // end get_vertex_fvalence()

/**************************************************************************
*
* Function: set_next_body_facet()
*
* Purpose: Set next link in body facet doubly linked list.
*/
void set_next_body_facet(
  facet_id f_id, // current facet
  facet_id ff_id // next facet
)
{ F_ELID(f_id,F_NEXT_BFACET_ATTR)[inverted(f_id)?1:0] = (ff_id) ; 
}

/**************************************************************************
*
* Function: set_prev_body_facet()
*
* Purpose: Set previous link in body facet doubly linked list
*/
void set_prev_body_facet(
   facet_id f_id, // current facet
   facet_id ff_id // previous facet
)
{ F_ELID(f_id,F_NEXT_BFACET_ATTR)[inverted(f_id)?3:2] = (ff_id) ; 
}

/**************************************************************************
*
* Function: get_next_body_facet()
*
* Purpose: Get next facet in body facet linked list.
*/
facet_id get_next_body_facet(facet_id f_id)
{ return  F_ELID(f_id,F_NEXT_BFACET_ATTR)[inverted(f_id)?1:0]; 
}

/**************************************************************************
*
* Function: get_prev_body_facet()
*
* Purpose: Get previous facet in body facet doubly linked list.
*/
facet_id get_prev_body_facet(facet_id f_id)
{ return  F_ELID(f_id,F_NEXT_BFACET_ATTR)[inverted(f_id)?3:2]; 
}

/**************************************************************************
*
* Function: set_body_volume()
*
* Purpose: Set value of body volume.
*/
void set_body_volume( 
  body_id b_id,
  REAL v,
  int mode /* NOSETSTAMP or SETSTAMP */
)
{ struct body *b = bptr(b_id);
  if ( !valid_id(b_id) ) return;
  b->volume = v; 
  b->abstotal = fabs(v); 
  if ( mode == SETSTAMP )
    b->voltimestamp = global_timestamp;
/*
  if ( web.representation == STRING)
  { facet_id f_id = get_body_facet(b_id));
    if ( valid_id(f_id) ) set_facet_area(f_id,v);
  }
*/
  if ( everything_quantities_flag )
  { struct gen_quant *q = GEN_QUANT(b->volquant);
    q->value = v;
  }
} // end set_body_volume()

/**************************************************************************
*
* Function: add_body_volume()
*
* Purpose:  Add to running total of body volume, using tree addition
*           for high accuracy.
*/
void add_body_volume(body_id b_id, REAL v)
{ /* uses binary tree add */ 
  
  struct body *b;
  if ( !valid_id(b_id) ) return;
  LOCK_ELEMENT(b_id);
  b = bptr(b_id);
  binary_tree_add(b->volume_addends,v);
  b->abstotal += fabs(v); 
  UNLOCK_ELEMENT(b_id);
} // end add_body_volume()

/**************************************************************************
*
* Function: add_body_volume_plain()
*
* Purpose: Add directly to body volume, e.g. volconst, not going through
*          tree addition, with updates of related quantities.
*/
void add_body_volume_plain(  /* with updates of related quantities */
  body_id b_id,
  REAL v
)
{ struct body *b;
  if ( !valid_id(b_id) ) return;
  b = bptr(b_id);
  LOCK_ELEMENT(b_id);
  b->volume += v;
  b->abstotal += fabs(v); 
  UNLOCK_ELEMENT(b_id);
  b->voltimestamp = global_timestamp;
/*
  if ( web.representation == STRING)
  { facet_id f_id = get_fe_facet(b->fe_id);
    if ( valid_id(f_id) ) set_facet_area(f_id,b->volume);
  }
*/
  if ( everything_quantities_flag )
  { struct gen_quant *q = GEN_QUANT(b->volquant);
    q->value = b->volume;
  } 
} // end add_body_volume_plain()

/**************************************************************************
*
* Function: add_body_abstotal() 
*
* Purpose: Add to total absolute sum of volume terms of body.
*/
void add_body_abstotal(body_id b_id, REAL v)
{ struct body *b;
  if ( !valid_id(b_id) ) return;
  b = bptr(b_id);
  b->abstotal += fabs(v); 
} // end add_body_abstotal()

/**************************************************************************
*
* Function: set_body_fixvol()
*
* Purpose: Set body target volume, and associated quantity target
*          if everything_quantities.
*/
void set_body_fixvol( 
  body_id b_id,
  REAL v
)
{ if ( valid_id(b_id) )
  { bptr(b_id)->fixvol = (v); 
    if ( everything_quantities_flag )
    { struct gen_quant *q = GEN_QUANT(get_body_volquant(b_id));
      q->target = v;
      if ( !(q->flags & Q_FIXED) )
      { q->flags &= ~(Q_INFO|Q_ENERGY|Q_CONSERVED);
        q->flags |= Q_FIXED;
      }
      if ( web.pressure_flag )
      { q = GEN_QUANT(get_body_ambquant(b_id));
        q->flags &= ~(Q_INFO|Q_FIXED|Q_CONSERVED);
        q->flags |= Q_ENERGY;
      }
    }
  }
  else
  { sprintf(errmsg,"fix body volume: illegal body %s.\n",ELNAME(b_id));
    kb_error(1307,errmsg,RECOVERABLE);
  }
} // end set_body_fixvol()

/**************************************************************************
*
* Function: new_vertex()
*
* Purpose: Create new vertex with given coordinates.
*/
vertex_id new_vertex(
  REAL *x,
  element_id parent /* for inherited stuff */
)
{ 
  int i;
  vertex_id v_id;
  REAL *y;

  v_id = new_element(VERTEX,parent,NULLID);
  if ( x )
  { y = get_coord(v_id);
    for ( i = 0 ; i < SDIM ; i++ ) y[i] = x[i];
  }
  set_vertex_edge(v_id,NULLID);
  set_vertex_facet(v_id,NULLID);

  set_v_global(v_id);

  /* interrupt conjugate gradient */
  if ( cg_hvector ) { myfree((char *)cg_hvector); cg_hvector = NULL; }

  return v_id;
} // end new_vertex()

/**************************************************************************
*
* Function: dup_vertex()
*
* Purpose: Create new vertex with duplicate properties of a given vertex.
*/
vertex_id dup_vertex(old_v)
vertex_id old_v;
{ 
  vertex_id v_id;
  struct vertex *v_id_p;

  v_id = new_element(VERTEX,NULLID,NULLID);
  v_id_p = vptr(v_id);
  memcpy((char *)&(v_id_p->attr),(char *)&(elptr(old_v)->attr),
              web.sizes[VERTEX] - ((char*)&(v_id_p->attr)-(char*)v_id_p));
  v_id_p->self_id = v_id;  /* restore new id */
  v_id_p->e_id = NULLID;
  return v_id;
} // end dup_vertex()

/**************************************************************************
*
* Function: new_edge()
*
* Purpose: Create new edge between two given vertices.
*/
edge_id new_edge(
  vertex_id tail_id,
  vertex_id head_id,
  element_id parent /* for inherited stuff */
)
{ 
  edge_id e_id;
  vertex_id v_id;
  REAL *x,*h,*t;
  int i,k;

  e_id = new_element(EDGE,parent,NULLID);
  set_edge_fe(e_id,NULLFACETEDGE);
  set_edge_color(e_id,DEFAULT_EDGE_COLOR);
  if ( valid_id(tail_id) && valid_id(head_id) )
  { set_edge_tailv(e_id,tail_id);
     set_edge_headv(e_id,head_id);
     if ( (web.modeltype == QUADRATIC) && valid_id(head_id) )
     { /* quadratic version; linear interpolation of midpoint */
        v_id = new_element(VERTEX,parent,NULLID);
        set_edge_midv(e_id,v_id);
        h = get_coord(head_id);
        t = get_coord(tail_id);
        x = get_coord(v_id);
        for ( i = 0 ; i < SDIM ; i++ )
          x[i] = (h[i] + t[i])/2.0;
     }
     else if ( (web.modeltype == LAGRANGE) && valid_id(head_id) )
     { /* Lagrange version; linear interpolation of points */
        vertex_id *v = get_edge_vertices(e_id);
        h = get_coord(head_id);
        t = get_coord(tail_id);
        for ( k = 1 ; k < web.lagrange_order ; k++ )
        { v[k] = new_element(VERTEX,parent,NULLID);
          set_attr(v[k],Q_MIDEDGE);
          set_vertex_edge(v[k],e_id);
          x = get_coord(v[k]);
          for ( i = 0 ; i < SDIM ; i++ )
             x[i] = (k*h[i] + (web.lagrange_order-k)*t[i])/web.lagrange_order;
        }
     }
  }
  return e_id;
} // end new_edge()

/**************************************************************************
*
* Function: dup_edge()
*
* Purpose: Create new edge as duplicate of given edge.
*/
edge_id dup_edge(edge_id old_e)
{ 
  edge_id e_id;
  struct edge *e_id_p,*old_e_p;
  vertex_id newmid; 

  e_id = new_element(EDGE,NULLID,NULLID);
  e_id_p = eptr(e_id);
  old_e_p = eptr(old_e);
  memcpy((char *)&(e_id_p->attr),(char *)&(old_e_p->attr),
              web.sizes[EDGE] - ((char*)&(e_id_p->attr)-(char*)e_id_p));
  e_id_p->self_id = e_id; /* restore new id */
  e_id_p->next_vedge[0] = NULLID;
  e_id_p->next_vedge[1] = NULLID;
  if ( web.modeltype == QUADRATIC )
  { newmid = new_vertex(NULL,old_e);
    set_edge_midv(e_id,newmid);
    set_attr(newmid,Q_MIDPOINT);
    set_vertex_edge(newmid,e_id);
  } 
  else if ( web.modeltype == LAGRANGE )
  { int i;
    vertex_id *v = get_edge_vertices(e_id);
    for ( i = 1 ; i < web.lagrange_order ; i++ )
      v[i] = new_vertex(NULL,e_id);
  } 
  if ( inverted(old_e) ) return inverse_id(e_id);
  return e_id;
} // end dup_edge()

/**************************************************************************
*
* Function: recalc_facet_area()
*
* Purpose: Recalculate area of a facet.
*/
void recalc_facet_area(facet_id f_id)
{
  if ( everything_quantities_flag )
    quantity_attribute(f_id,default_area_quant_num);
  else if ( web.representation == SOAPFILM )
    (*calc_facet_energy)(f_id,AREA_ONLY);
} // end recalc_facet_area()

/**************************************************************************
*
* Function: new_facet()
*
* Purpose: Create new facet.
*/
facet_id new_facet()
{ 
  facet_id f_id;

  f_id = new_element(FACET,NULLID,NULLID);
  set_facet_color(f_id,DEFAULT_FACET_COLOR);
  set_facet_density(f_id,1.0);
  if ( opacity_attr ) 
    *(REAL*)(get_extra(f_id,opacity_attr)) = 1.0;
  return f_id;
}

/**************************************************************************
*
* Function: dup_facet()
*
* Purpose: Create new facet as duplicate of given facet.
*/
facet_id dup_facet(facet_id old_f)
{ 
  facet_id f_id;
  struct facet *f_id_p;
  body_id b_id,*fb;
  int sign = inverted(old_f);

  old_f = positive_id(old_f);  /* since new facet id will be positive */
  
  f_id = new_element(FACET,NULLID,NULLID);
  f_id_p = fptr(f_id);
  memcpy((char *)&(f_id_p->attr),(char *)&(elptr(old_f)->attr),
              web.sizes[FACET] - ((char*)&(f_id_p->attr)-(char*)f_id_p));
  f_id_p->self_id = f_id; /* restore new id */
  set_attr(f_id,NEWELEMENT);

  /* update body facet lists */
  if ( web.skel[BODY].count )
  { fb = F_ELID(f_id,F_BODY_LIST_ATTR);
    fb[0] = fb[1] = NULLID; /* so set_facet_body works */
    fb = F_ELID(f_id,F_NEXT_BFACET_ATTR);
    fb[0] = fb[1] = fb[2] = fb[3] = NULLID;
    b_id = get_facet_body(old_f);
    if ( valid_id(b_id) )
      set_facet_body(f_id,b_id);
    b_id = get_facet_body(inverse_id(old_f));
    if ( valid_id(b_id) )
      set_facet_body(inverse_id(f_id),b_id);
  }

  return sign ? inverse_id(f_id) : f_id;
} // end dup_facet()

/**************************************************************************
*
* Function: new_body()
*
* Purpose: Create a new body.
*/
body_id new_body()
{ int two = 2;
  int four = 4;
  body_id b_id;

  expand_attribute(FACET,F_BODY_LIST_ATTR,&two);
  expand_attribute(FACET,F_NEXT_BFACET_ATTR,&four);
  b_id = new_element(BODY,NULLID,NULLID);
  set_body_facet(b_id,NULLID);
  set_attr(b_id,WANT_CENTEROFMASS);
  if ( everything_quantities_flag ) 
      convert_new_body_to_quantity(b_id);
  web.bodycount++;
  return b_id;
} // end new_body()

/**************************************************************************
*
* Function: dup_body()
*
* Purpose:  Create new body as duplicate of given body.
*/
body_id dup_body(body_id old_b)
{ 
  body_id b_id;
  struct body *b_id_p;

  b_id = new_element(BODY,NULLID,NULLID);
  b_id_p = bptr(b_id);
  memcpy((char *)&(b_id_p->attr),(char *)&(elptr(old_b)->attr),
              web.sizes[BODY] - ((char*)&(b_id_p->attr)-(char*)b_id_p));
  b_id_p->self_id = b_id; /* restore new id */
  set_body_facet(b_id,NULLID);
  if ( everything_quantities_flag ) 
      convert_new_body_to_quantity(b_id);
  web.bodycount++;
  return b_id;
} // end dup_body()

/**************************************************************************
*
* Function: new_facetedge()
*
* Purpose: Create new facetedge linking given facet and edge.
*/
facetedge_id new_facetedge(
  facet_id f_id,
  edge_id e_id
)
{ 
  facetedge_id fe_id;
 
  fe_id = new_element(FACETEDGE,NULLID,NULLID);
  set_fe_edge(fe_id,e_id);
  set_fe_facet(fe_id,f_id);
  set_prev_edge(fe_id,NULLFACETEDGE);
  set_next_edge(fe_id,NULLFACETEDGE);
  set_prev_facet(fe_id,NULLFACETEDGE);      
  set_prev_facet(fe_id,NULLFACETEDGE);
  #ifndef MPI_EVOLVER
  { vertex_id headv,tailv;
    tailv = get_edge_tailv(e_id);  
    headv = get_edge_headv(e_id);
    if ( !valid_id(get_vertex_edge(tailv)) )
      set_vertex_edge(tailv,e_id);
    if ( !valid_id(get_vertex_edge(headv)) )
      set_vertex_edge(headv,e_id);
  }
  #endif

  if ( web.representation==STRING && everything_quantities_flag )
  { /* attach volume quantities */
     body_id b_id;
     b_id = get_facet_body(f_id);
     if ( valid_id(b_id) )
     { if ( same_sign(f_id,e_id) )
          apply_method_num(e_id,get_body_volmeth(b_id));
        else
          apply_method_num(inverse_id(e_id),get_body_volmeth(b_id));
     }
     b_id = get_facet_body(inverse_id(f_id));
     if ( valid_id(b_id) )
     { if ( same_sign(f_id,e_id) )
          apply_method_num(inverse_id(e_id),get_body_volmeth(b_id));
        else
          apply_method_num(e_id,get_body_volmeth(b_id));
     }
  }
  return fe_id;
} // end new_facetedge()

/**************************************************************************
*
* Function: set_fe_facet()
*
* Purpose: Set the facet link of a facetedge.
*/
void set_fe_facet(
  facetedge_id fe_id,
  facet_id f_id
)
{ facet_id oldf = get_fe_facet(fe_id);

  feptr(fe_id)->fe_facet_id = inverted(fe_id) ? inverse_id(f_id) : f_id;

  if ( web.representation==STRING && everything_quantities_flag &&
        !equal_id(oldf,f_id) )
  { /* detach old volume quantity and attach new volume quantity */    
    edge_id e_id = get_fe_edge(fe_id);
    check_edge_vol_methods(e_id);
  }
} // end set_fe_facet()

/*************************************************************************
*
* Function: check_edge_vol_methods()
*
* Purpose: When everything_quantities, check that edge has proper
*          body volume quantities.  To be called after topology
*          changes regarding an edge.  Will fix up edge if wrong.
*          Assumes edge valence at most 2.
*/
void check_edge_vol_methods(edge_id e_id)
{
  int *instlist;
  int meth_offset; 
  struct edge *e_ptr;
  facetedge_id start_fe,fe;
  int should_be[2];
  int should_count = 0;
  facet_id f_id,ff_id;
  body_id b_id,bb_id;
  int i,j;

  start_fe = get_edge_fe(e_id);
  if ( !valid_id(start_fe) )
    should_count = 0;
  else
  { fe = get_next_facet(start_fe);
    if ( equal_id(fe,start_fe) )
    { // valence 1
      f_id = get_fe_facet(start_fe);
      b_id = get_facet_body(f_id);
      if ( valid_id(b_id) )
      { should_be[0] = get_body_volmeth(b_id);
        should_count = 1;
      }
      b_id = get_facet_body(inverse_id(f_id));
      if ( valid_id(b_id) )
      { should_be[0] = -get_body_volmeth(b_id);
        should_count = 1;
      }
    }
    else // valence 2
    { int sign1 = 1, sign2 = 1;
      f_id = get_fe_facet(start_fe);
      b_id = get_facet_body(f_id);
      if ( !valid_id(b_id) )
      { b_id = get_facet_body(inverse_id(f_id));
        sign1 = -1;
      }
      ff_id = get_fe_facet(fe);
      bb_id = get_facet_body(ff_id);
      if ( !valid_id(bb_id) )
      { bb_id = get_facet_body(inverse_id(ff_id));
        sign2 = -1;
      }
      if ( valid_id(b_id) && valid_id(bb_id) )
      { if ( !equal_id(b_id,bb_id) )
        { should_be[0] = sign1*get_body_volmeth(b_id);
          should_be[1] = sign2*get_body_volmeth(bb_id);
          should_count = 2;
        }
      }
      else if ( valid_id(b_id) )
      {  should_be[0] = sign1*get_body_volmeth(b_id);
         should_count = 1;
      }
      else if ( valid_id(bb_id) )
      { should_be[0] = sign2*get_body_volmeth(bb_id);
        should_count = 1;
      }
      else
        should_count = 0;
    }
  }

  meth_offset = get_meth_offset(EDGE); 
  e_ptr = (struct edge *)elptr(e_id);
  instlist = (int*)((char*)e_ptr + meth_offset); 
 
  // see if any unwanted
  for ( i = 0 ; i < (int)e_ptr->method_count ; i++ )
  { struct method_instance *mi = METH_INSTANCE(instlist[i]);
    if ( mi->flags & BODY_INSTANCE )
    { int found = 0;
      for ( j = 0 ; j < should_count ; j++ )
        if ( should_be[j] == instlist[i] )
        { found = 1;
          should_be[j] = should_be[--should_count]; // remove from list
          j--;
        }
      if ( !found ) // remove
      { instlist[i] = instlist[--e_ptr->method_count]; 
        i--;
      }
    }
  }

  // remaining ones in list need to be added
  for ( j = 0 ; j < should_count ; j++ )
    apply_method_num(e_id,should_be[j]);

} // end check_edge_vol_methods()

/**************************************************************************
*
* Function: get_edge_length()
*
* Purpose: Calculate and return the length of an edge.
*/
REAL get_edge_length(e_id)
edge_id e_id;
{
  calc_edge(e_id);
  return (eptr(e_id)->length);
} // end get_edge_length()

/**************************************************************************
*
* Function: get_facet_pressure()
*
* Purpose: Return difference of pressure across a facet.
*/
REAL get_facet_pressure(f_id)
facet_id f_id;
{ 
  return  (get_body_pressure(get_facet_body(f_id)) - 
        get_body_pressure(get_facet_body(facet_inverse(f_id))));
} // end get_facet_pressure()


/********************************************************************
*
* Function: compare_vertex_attr(va,vb)
* 
* Purpose: Compare constraints of vertex and vertex.
*
* Return value: 0 INCOMPARABLE  Incomparable
*               1 A_SUB_B Vertex proper subset of vertex
*               2 A_EQ_B Vertex equal vertex
*               3 A_SUPER_B Vertex superset of vertex
*/

int compare_vertex_attr(vertex_id va, vertex_id vb)
{ int i,j;
  conmap_t * con1,*con2;
  int bdry1,bdry2;
  int same = 0;

  con1 = get_v_constraint_map(va);
  con2 = get_v_constraint_map(vb);

  bdry1 = get_vertex_boundary_num(va);
  bdry2 = get_vertex_boundary_num(vb);

  /* check boundaries */
  if ( bdry1 )
  { if ( bdry2 )
    { if ( bdry1 == bdry2 ) 
         return A_EQ_B;
      else return INCOMPARABLE;
    }
    else if ( con2[0] )
      return INCOMPARABLE;
    else 
      return A_SUPER_B;
  }
  else if ( bdry2 )
  { if ( con1[0] )
      return INCOMPARABLE;
    else
    return A_SUB_B;
  }
 
  /* check constraints */
  for ( i = 1 ; i <= (int)con2[0] ; i++ )
  { for ( j = 1 ; j <= (int)con1[0] ; j++ )
      if ( (con1[j]&CONMASK) == (con2[i]&CONMASK) )
      { same++;
        break;
      }
  }

  if ( (same==(int)con1[0]) && (same==(int)con2[0]) )
    return A_EQ_B;
  if ( (same==(int)con1[0]) && (same < (int)con2[0]) )
    return A_SUB_B;
  if ( (same < (int)con1[0]) && (same==(int)con2[0]) )
    return A_SUPER_B;
  return INCOMPARABLE;
} // end compare_vertex_attr()

/********************************************************************
*
* Function: compare_vertex_edge_attr(va,eb)
* 
* Purpose: Compare constraints of vertex and edge.
*
* Return value: 0 INCOMPARABLE  Incomparable
*               1 A_SUB_B Vertex proper subset of edge
*               2 A_EQ_B Vertex equal edge
*               3 A_SUPER_B Vertex superset of edge
*/

int compare_vertex_edge_attr(vertex_id va,edge_id eb)
{ int i,j;
  conmap_t * con1,*con2;
  int bdry1,bdry2;
  int same = 0;

  con1 = get_v_constraint_map(va);
  con2 = get_e_constraint_map(eb);

  bdry1 = get_vertex_boundary_num(va);
  bdry2 = get_edge_boundary_num(eb);

  /* check boundaries */
  if ( bdry1 )
  { if ( bdry2 )
    { if ( bdry1 == bdry2 ) 
         return A_EQ_B;
      else return INCOMPARABLE;
    }
    else if ( con2[0] )
      return INCOMPARABLE;
    else 
      return A_SUPER_B;
  }
  else if ( bdry2 )
  { if ( con1[0] )
      return INCOMPARABLE;
    else
    return A_SUB_B;
  }
 
  /* check constraints */

  for ( i = 1 ; i <= (int)con2[0] ; i++ )
  { for ( j = 1 ; j <= (int)con1[0] ; j++ )
      if ( (con1[j]&CONMASK) == (con2[i]&CONMASK) )
      { same++;
        break;
      }
  }

  if ( (same==(int)con1[0]) && (same==(int)con2[0]) )
    return A_EQ_B;
  if ( (same==(int)con1[0]) && (same < (int)con2[0]) )
    return A_SUB_B;
  if ( (same < (int)con1[0]) && (same==(int)con2[0]) )
    return A_SUPER_B;
  return INCOMPARABLE;

} // end compare_vertex_edge_attr()

/********************************************************************
*
* Function: compare_edge_attr(ea,eb)
* 
* Purpose: Compare constraints of edge and edge.
*
* Return value: 0 INCOMPARABLE  Incomparable
*               1 A_SUB_B edge proper subset of edge
*               2 A_EQ_B edge equal edge
*               3 A_SUPER_B edge superset of edge
*/

int compare_edge_attr(edge_id ea, edge_id eb)
{ int i,j;
  int fixa = get_eattr(ea) & FIXED;
  int fixb = get_eattr(eb) & FIXED;
  conmap_t * con1,*con2;
  int bdry1,bdry2;
  int same = 0;

  con1 = get_e_constraint_map(ea);
  con2 = get_e_constraint_map(eb);

  bdry1 = get_edge_boundary_num(ea);
  bdry2 = get_edge_boundary_num(eb);

  /* check boundaries */
  if ( bdry1 )
  { if ( bdry2 )
    { if ( bdry1 == bdry2 ) 
         return A_EQ_B;
      else return INCOMPARABLE;
    }
    else if ( con2[0] )
      return INCOMPARABLE;
    else 
      return A_SUPER_B;
  }
  else if ( bdry2 )
  { if ( con1[0] )
      return INCOMPARABLE;
    else
      return A_SUB_B;
  }
 
  /* check constraints */

  for ( i = 1 ; i <= (int)con2[0] ; i++ )
  { for ( j = 1 ; j <= (int)con1[0] ; j++ ) 
      if ( (con1[j]&CONMASK) == (con2[i]&CONMASK) )
      { same++;
        break;
      }
  }

  if ( (same==(int)con1[0]) && (same==(int)con2[0]) && !fixa && !fixb )
    return A_EQ_B;
  if ( (same==(int)con1[0]) && (same < (int)con2[0]) && !fixa )
    return A_SUB_B;
  if ( (same < (int)con1[0]) && (same==(int)con2[0]) && !fixb )
    return A_SUPER_B;
  return INCOMPARABLE;
} // end compare_edge_attr()


/********************************************************************
*
* Function: compare_edge_facet_attr(ea,fb)
* 
* Purpose: Compare constraints of edge and facet.
*
* Return value: 0 INCOMPARABLE  Incomparable
*               1 A_SUB_B Edge proper subset of facet
*               2 A_EQ_B Edge equal facet
*               3 A_SUPER_B Edge superset of facet
*/

int compare_edge_facet_attr(edge_id ea,facet_id fb)
{ int i,j;
  conmap_t * con1,*con2;
  int fixa = get_eattr(ea) & FIXED;
  int fixb = get_fattr(fb) & FIXED;
  int same = 0;
  int bdry1,bdry2;

  con1 = get_e_constraint_map(ea);
  con2 = get_f_constraint_map(fb);

  bdry1 = get_edge_boundary_num(ea);
  bdry2 = get_facet_boundary_num(fb);

  /* check boundaries */
  if ( bdry1 )
  { if ( bdry2 )
    { if ( bdry1 == bdry2 ) 
         return A_EQ_B;
      else return INCOMPARABLE;
    }
    else if ( con2[0] )
      return INCOMPARABLE;
    else 
      return A_SUPER_B;
  }
  else if ( bdry2 )
  { if ( con1[0] )
      return INCOMPARABLE;
    else
    return A_SUB_B;
  }
 
  /* check constraints */
  for ( i = 1 ; i <= (int)con2[0] ; i++ )
  { for ( j = 1 ; j <= (int)con1[0] ; j++ )
      if ( (con1[j]&CONMASK) == (con2[i]&CONMASK) )
      { same++;
        break;
      }
  }

  if ( (same==(int)con1[0]) && (same==(int)con2[0])&& !fixa && !fixb )
    return A_EQ_B;
  if ( (same==(int)con1[0]) && (same < (int)con2[0]) && !fixa )
    return A_SUB_B;
  if ( (same < (int)con1[0]) && (same==(int)con2[0]) && !fixb )
    return A_SUPER_B;
  return INCOMPARABLE;

} // end compare_edge_facet_attr()

/**************************************************************
*
*  Function:  equal_constr()
*
*  Purpose: See if two elements have the same set of constraints.
*
*/

int equal_constr(
  element_id id1,
  element_id id2
)
{ int i,j;
  conmap_t * con1=NULL,*con2=NULL;

  switch ( id_type(id1) )
     {
        case VERTEX: 
                         con1        = get_v_constraint_map(id1);
                         break;

        case EDGE  : 
                         con1        = get_e_constraint_map(id1);
                         break;

        case FACET : 
                         con1        = get_f_constraint_map(id1);
                         break;
     }


  switch ( id_type(id2) )
     {
        case VERTEX: 
                         con2        = get_v_constraint_map(id2);
                         break;

        case EDGE  :
                         con2        = get_e_constraint_map(id2);
                         break;

        case FACET :
                         con2        = get_f_constraint_map(id2);
                         break;
     }
  if ( con2[0] != con1[0] ) return 0;
  for ( i = 1 ; i <= (int)con1[0] ; i++ )
  { for ( j = 1 ; j <= (int)con2[0] ; j++ )
        if ( (con1[i]&CONMASK) == (con2[j]&CONMASK) ) break;
     if ( j > (int)con2[0] ) return 0;
  }
  return 1;
} // end equal_constr()

/*************************************************************************
*
*  function: add_attribute()
*
*  purpose: add extra attribute to an element type
*
*  return:  index of attribute
*/

int add_attribute(
  int e_type, /* VERTEX ,... */
  char *name,
  int attr_type, /* REAL_TYPE or INTEGER_TYPE or ULONG_TYPE etc */
  int dim, /* number of dimensions, 0 for scalar */
  int *dims, /* sizes of dimensions, NULL for all sizes 0 */
            /* Note: scalar still needs size of 0 or 1 */ 
  int dumpflag, /* whether appears in dump file */
  struct expnode *code, /* nonnull for function attribute */
  int mpi_propagate /* MPI_PROPAGATE to broadcast */
)
{ int newsize,newcount=0;
  struct extra *ex;
  int oldsize;
  int att_inx;

  att_inx = find_attribute(e_type,name);
  if ( att_inx >= 0 )
    return att_inx;

#ifdef MPI_EVOLVER
  if ( (this_task == MASTER_TASK) && (mpi_propagate==MPI_PROPAGATE) && !mpi_initialization_flag )
  { struct mpi_command message;
    int i;

    if ( code )
    { kb_error(4525,"MPI Evolver doesn't do code attributes yet.\n",
          RECOVERABLE);
    }
    message.cmd = mpi_ADD_ATTRIBUTE;
    strncpy(message.name,name,MPI_NAME_SIZE);
    message.i = attr_type;
    message.type = e_type;
    message.mode = dumpflag;
	message.size = dims ? 1 : 0; /* just using as indicator */
    message.count = dim;
	if ( dim )
      for (i = 0 ; i < dim ; i++ )
		  message.data[i] = dims ? dims[i] : 0;
	else 
		message.data[0] = dims ? dims[0] : 0;
    MPI_Bcast(&message,sizeof(struct mpi_command),MPI_BYTE,MASTER_TASK,
        MPI_COMM_WORLD);
  }
#endif

  if ( web.skel[e_type].extra_count >= web.skel[e_type].maxextra-1 )
  { web.skel[e_type].dy_extras = 
       dy_realloc(web.skel[e_type].dy_extras,
       (web.skel[e_type].maxextra+10),sizeof(struct extra));
    web.skel[e_type].maxextra += 10;
  }

  /* expand space */ 
  /* get old used space, minus any padding, and pad to proper size */
  if ( web.skel[e_type].extra_count > 0 )
  { ex = EXTRAS(e_type) + web.skel[e_type].extra_count-1;
    oldsize = ex->offset + ex->array_spec.datacount*datatype_size[ex->type];
  }
  else oldsize = web.sizes[e_type];
  if  (oldsize % datatype_size[attr_type])
    oldsize += datatype_size[attr_type] - (oldsize % datatype_size[attr_type]);

  ex = EXTRAS(e_type) + web.skel[e_type].extra_count;

  if ( dim == 0 ) newcount = dims ? dims[0] : 0;
  else if ( dim > MAXARRAYDIMS )
  { sprintf(errmsg,"Extra attribute \"%s\" has %d dimensions, exceeding limit of %d.\n", name,dim,MAXARRAYDIMS);
    kb_error(2510,errmsg,RECOVERABLE);
  }
  else if ( dims == NULL ) newcount = 0;
  else
  { int i;
    for ( i = 0, newcount = 1 ; i < dim ; i++ )
    { newcount *= dims[i];
      ex->array_spec.sizes[i] = dims[i];
    }
  }
  if ( (newcount > 0) || (oldsize > web.sizes[e_type]) )
  { newsize =  newcount*datatype_size[attr_type];
    expand(e_type,oldsize + newsize); 
  }
  strncpy(ex->name,name,ATTR_NAME_SIZE);
  ex->type = attr_type;
  ex->array_spec.datatype = attr_type;
  ex->array_spec.itemsize = datatype_size[attr_type];
 
  ex->offset = oldsize;
  ex->array_spec.datacount = newcount;
  ex->array_spec.dim = dim;
  if ( dim > 0 ) ex->flags |= DIMENSIONED_ATTR;
  if ( code ) ex->code = *code;
  if ( dumpflag ) ex->flags |= DUMP_ATTR;

  if ( stricmp(name,EXTRA_BDRY_NAME) == 0 )
  { extra_bdry_attr = web.skel[e_type].extra_count;
    if ( ex->type != INTEGER_TYPE )
      kb_error(2842,"Attribute extra_boundary must be of type integer.\n",
         RECOVERABLE);
    if ( e_type != VERTEX )
      kb_error(2843,"Attribute extra_boundary should be vertex attribute.\n",
         RECOVERABLE);
  }
  else if ( stricmp(name,EXTRA_BDRY_PARAM_NAME) == 0 )
  { extra_bdry_param_attr = web.skel[e_type].extra_count;
    if ( ex->type != REAL_TYPE )
      kb_error(3369,"Attribute extra_boundary_param must be of type real.\n",
         RECOVERABLE);
    if ( e_type != VERTEX )
      kb_error(2845,
        "Attribute extra_boundary_param should be vertex attribute.\n",
         RECOVERABLE);
  }
  web.skel[e_type].extra_count++;
  return web.skel[e_type].extra_count - 1; /* index */
} // end add_attribute()


/*************************************************************************
*
*  function: expand_attribute()
*
*  purpose: enlarge space for attribute of an element type
*              or shrink it.
*/

void expand_attribute(
  int e_type, /* VERTEX ,... */
  int attr_num, /* number of attribute */
  int *newsizes /* new numbers of components, necessary also for scalar */
)
{ int newsize;  
  int diff;  /* difference between old and new total sizes */
  struct extra *ex;
  int chunksize,offset,available,needed;
  int pointercount=0;
  char *spot;
  element_id id;
  int k,d,n,dsize,dest,inx,blocksize;
  char *temp=NULL; /* for shuffling higher dim entries */

  ex = EXTRAS(e_type) + attr_num;

#ifdef MPI_EVOLVER
  if ( (this_task == MASTER_TASK) && !mpi_initialization_flag )
  { struct mpi_command message;
    int i;
    message.cmd = mpi_EXPAND_ATTRIBUTE;
    message.i = attr_num;
    message.type = e_type;
    message.count = ex->array_spec.dim;
	if ( ex->array_spec.dim )
      for (i = 0 ; i < ex->array_spec.dim ; i++ )
        message.data[i] = newsizes[i];
	else message.data[0] = newsizes[0];
    MPI_Bcast(&message,sizeof(struct mpi_command),MPI_BYTE,MASTER_TASK,
        MPI_COMM_WORLD);
  }
#endif

  dsize = ex->array_spec.itemsize;

  if ( ex->array_spec.dim == 0 ) newsize = newsizes[0];
  else
  { pointercount = 1;
    for ( newsize = 1, k = 0 ; k < ex->array_spec.dim ; k++ )
    { newsize *= newsizes[k];
      if ( k < ex->array_spec.dim - 1 )
        pointercount *= newsizes[k];
    }
  }
  

  if ( (ex->array_spec.dim <= 1) && (newsize == ex->array_spec.datacount) ) 
     return;

  /* expand or contract space */
  /* see how much extra space is needed */
  if ( attr_num < web.skel[e_type].extra_count-1 )
         available = ex[1].offset - ex[0].offset;
  else available = web.sizes[e_type] - ex->offset;
  needed = newsize*datatype_size[ex->type]+pointercount*sizeof(REAL*);
  if ( ex->array_spec.dim >= 2 ) 
     temp = (char*)temp_calloc(needed,1);
  if ( needed > available )
  { /* expand */
    /* check alignment of following fields */
    diff = needed - available;
    for ( k = attr_num+1 ; k < web.skel[e_type].extra_count ; k++ )
      while ( diff % datatype_size[EXTRAS(e_type)[k].type] )
        diff++;
    expand(e_type,web.sizes[e_type] + diff); 
    /* move stuff above */
    if ( attr_num < web.skel[e_type].extra_count-1 )
      offset = EXTRAS(e_type)[attr_num+1].offset;
    else offset = web.sizes[e_type] - diff;
    chunksize = web.sizes[e_type] - offset - diff;
    if ( chunksize || ( ex->array_spec.dim >= 2) )
    { element_id sentinel;
      id = NULLID; 
      if ( web.skel[e_type].count > 0 ) 
        while ( generate_all(e_type,&id,&sentinel) )
        { 
          spot = (char*)elptr(id) + offset;
          kb_memmove(spot + diff,spot,chunksize);
          if ( ex->array_spec.dim >= 2 )
          { /* entry shuffle via temp */
            char *old = (char*)elptr(id) + EXTRAS(e_type)[attr_num].offset;
            for ( n = 0 ; n < ex->array_spec.datacount ; n++ )
            { /* figure out indices and new spot */
              int oldinx = n;
              for ( d = ex->array_spec.dim-1, dest = 0, blocksize = 1 ; d >= 0 ; d-- )
              { inx = oldinx % ex->array_spec.sizes[d];
                if ( inx >= newsizes[d] ) goto skipentry; 
                dest += blocksize*inx;
                blocksize *= newsizes[d];
                oldinx = oldinx/ex->array_spec.sizes[d];
              }
              kb_memmove(temp+dest*dsize,old+n*dsize,dsize);
skipentry:    ;
            }
            kb_memmove(old,temp,needed);
          }
          else
            memset(spot,0,diff);
        }
    }
    for ( k = attr_num+1 ; k < web.skel[e_type].extra_count ; k++ )
         EXTRAS(e_type)[k].offset += diff;
  }
  else if ( needed < available )
  { /* maybe shrink */
     /* check alignment of following fields */
     diff = available - needed;
     for ( k = attr_num+1 ; k < web.skel[e_type].extra_count ; k++ )
        while ( diff % datatype_size[EXTRAS(e_type)[k].type] )
          diff--;
     /* move stuff above */
     if ( attr_num < web.skel[e_type].extra_count-1 )
       offset = EXTRAS(e_type)[attr_num+1].offset;
     else offset = web.sizes[e_type];
     chunksize = web.sizes[e_type] - offset;
     if ( chunksize || (ex->array_spec.dim >= 2) )
     { element_id sentinel;
       id = NULLID;
       if ( web.skel[e_type].count > 0 ) 
        while ( generate_all(e_type,&id,&sentinel) )
        {
          if ( ex->array_spec.dim >= 2 )
          { /* entry shuffle via temp */
            char *old = (char*)elptr(id)+EXTRAS(e_type)[attr_num].offset;
            for ( n = 0 ; n < ex->array_spec.datacount ; n++ )
            { /* figure out indices and new spot */
              int oldinx = n;
              for ( d = ex->array_spec.dim-1, dest = 0, blocksize = 1 ; d >= 0 ; d-- )
              { inx = oldinx % ex->array_spec.sizes[d];
                if ( inx >= newsizes[d] ) goto skipentry2; 
                dest += blocksize*inx;
                blocksize *= newsizes[d];
                oldinx = oldinx/ex->array_spec.sizes[d];
              }
              kb_memmove(temp+dest*dsize,old+n*dsize,dsize);
skipentry2:    ;
            }
            kb_memmove(old,temp,needed);
          }
          spot = (char*)elptr(id) + offset;
          kb_memmove(spot - diff,spot,chunksize);
        }
     }
     for ( k = attr_num+1 ; k < web.skel[e_type].extra_count ; k++ )
         EXTRAS(e_type)[k].offset -= diff;
     expand(e_type,web.sizes[e_type] - diff); 
  }
  if ( ex->array_spec.dim >= 2 ) temp_free(temp);
  ex->array_spec.datacount = newsize;
  for ( n = 0 ; n < ex->array_spec.dim ; n++ )
    ex->array_spec.sizes[n] = newsizes[n];
  parallel_update_flag[e_type] = 1;

#ifdef MPI_EVOLVER
  mpi_export_voffset = EXTRAS(VERTEX)[web.mpi_export_attr[VERTEX]].offset;
  mpi_export_eoffset = EXTRAS(EDGE)[web.mpi_export_attr[EDGE]].offset;
  mpi_export_foffset = EXTRAS(FACET)[web.mpi_export_attr[FACET]].offset;
  mpi_export_boffset = EXTRAS(BODY)[web.mpi_export_attr[BODY]].offset;
  mpi_export_feoffset =
          EXTRAS(FACETEDGE)[web.mpi_export_attr[FACETEDGE]].offset; 
#endif
} // end expand_attribute()

/*************************************************************************
*
*  function: find_attribute()
*
*  purpose: find extra attribute by name, if it exists.
*  return: index number if found, -1 if not.
*/

int find_attribute(
  int etype,
  char *name
)
{ struct extra *ex;
  int n;
  ex = EXTRAS(etype);
  for ( n = 0 ; n < web.skel[etype].extra_count ; n++,ex++ )
    if ( stricmp(ex->name,name) == 0 ) break;
  if ( n == web.skel[etype].extra_count )
     return -1;
  return n;
} // end find_attribute()

/**************************************************************************
*
* function: find_extra()
*
* purpose: return index of named attribute, searching all element types.
*             return -1 if not found.
*/
int find_extra(
  char *name,
  int *etype /* for returning element type */
)
{ int el_type,qnum,n;
  struct extra *ex;

  for ( el_type = VERTEX, qnum = -1 ; el_type <= FACETEDGE ; el_type++ )
  { ex = EXTRAS(el_type);
    for ( n = 0 ; n < web.skel[el_type].extra_count ; n++,ex++ )
      if ( stricmp(ex->name,name) == 0 )
      {*etype = el_type;qnum = n;break;}
  }
  return qnum;
} // end find_extra()

/***************************************************************************
  Constraint handling routines
****************************************************************************/

/* Methodology:
    Constraint map is array of conmap_t.
    First entry is number of constraints.
    Then follow constraint numbers, with high bit CON_HIT_BIT set
      if constraint is hit.
    Allocated as an extra attribute if needed.
*/

conmap_t nullcon[2]; /* default empty list */

/**************************************************************************
*
* Function: set_v_global()
*
* Purpose: Set vertex to be on all global level-set constraints.
*/
void set_v_global(vertex_id v_id)
{ int k;
  for ( k = 0 ; k < web.con_global_count ; k++ )
     set_v_constraint_map(v_id,web.con_global_map[k]);
} // end set_v_global()

/**************************************************************************
*
* Function: set_v_conmap()
*
* Purpose: Add vertex level-set constraints from those given in "map"
*/
void set_v_conmap(
  vertex_id v_id,
  conmap_t *map
)
{ int k, m=(int)map[0];
  for ( k = 1 ; k <= m ; k++ )
    set_v_constraint_map(v_id,map[k]);
  map = get_v_constraint_map(v_id);
  if ( map[0] == 0 ) unset_attr(v_id,CONSTRAINT);
} // end set_v_conmap()

/**************************************************************************
*
* Function: set_e_conmap()
*
* Purpose: Add edge level-set constraints from those given in "map"
*/
void set_e_conmap(e_id,map)  
edge_id e_id;
conmap_t *map;
{ int k, m=(int)map[0];
  for ( k = 1 ; k <= m ; k++ )
    set_e_constraint_map(e_id,map[k]);
  map = get_e_constraint_map(e_id);
  if ( map[0] == 0 ) unset_attr(e_id,CONSTRAINT);
} // end set_e_conmap()

/**************************************************************************
*
* Function: set_f_conmap()
*
* Purpose: Add facet level-set constraints from those given in "map"
*/
void set_f_conmap(  
  facet_id f_id,
  conmap_t *map
)
{ int k, m=(int)map[0];
  for ( k = 1 ; k <= m ; k++ )
    set_f_constraint_map(f_id,map[k]);
  map = get_f_constraint_map(f_id);
  if ( map[0] == 0 ) unset_attr(f_id,CONSTRAINT);
} // end set_f_conmap()

/**************************************************************************
*
* Function: set_v_constraint_map()
*
* Purpose: Set an individual level-set constraint on a vertex.
*/
void set_v_constraint_map( 
  vertex_id v_id,
  int n   /* constraint number */
)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;
  struct constraint *constr;
  int k,j;
  int four = 4;

  n &= CONMASK;
  if ( maxcon == 0 )
     expand_attribute(VERTEX,V_CONSTR_LIST_ATTR,&four);
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
     if ( (map[k] & CONMASK) == ((conmap_t)n & CONMASK) ) return;
  if ( k >= maxcon )
  { int newmax;
    newmax = maxcon+4;
    expand_attribute(VERTEX,V_CONSTR_LIST_ATTR,&newmax);
    map = get_v_constraint_map(v_id);
  }  
  constr = get_constraint(n);
  map[k] = (conmap_t)n; 
  if ( !(constr->attr & (NONPOSITIVE|NONNEGATIVE) )) 
    map[k] |= CON_HIT_BIT;
  map[0]++; /* counter */

  set_attr(v_id,CONSTRAINT);
  if ( (constr->attr & CON_ENERGY) && (web.representation == STRING) )
  { set_attr(v_id, BDRY_ENERGY);
     if ( everything_quantities_flag )
        apply_method_num(v_id,constr->energy_method);
  }
  if ( (constr->attr & CON_CONTENT) && (web.representation == STRING) )
  { 
    set_attr(v_id, BDRY_CONTENT);

    if ( everything_quantities_flag )
    { edge_id e_id,first_e;
      int max_rank,min_rank;
        
      min_rank = MAXINT; max_rank = 0;
      for ( j = 1 ; j <= (int)map[0] ; j++ )
      { struct constraint *c;
        if ( !(map[j] & CON_HIT_BIT) ) continue;
        c = get_constraint(map[j]);
        if ( c->content_rank < min_rank ) min_rank = c->content_rank;
        if ( c->content_rank > max_rank ) max_rank = c->content_rank;
      }

      first_e = e_id = get_vertex_edge(v_id);
      if ( valid_id(e_id) && !(get_eattr(e_id) & NONCONTENT) )
      do
      { 
       body_id b_id;
       facetedge_id first_fe,fe;
       first_fe = fe = get_edge_fe(e_id);
       if ( valid_id(fe) )  do
       { facet_id f_id = get_fe_facet(fe);
         fe = get_next_facet(fe);
         if ( !valid_id(f_id) ) continue;
         b_id = get_facet_body(f_id);
         
         if ( valid_id(b_id)  && 
           ( (!inverted(f_id) && constr->content_rank >= max_rank) 
               || (inverted(f_id) && constr->content_rank <= min_rank)))
         { int methnum;
           methnum = create_body_constraint_content_method(b_id,n);
           attach_method_num(get_body_volquant(b_id),methnum);
           apply_method_num(inverse_id(v_id),methnum);
         }

         b_id = get_facet_body(inverse_id(f_id));         
         if ( valid_id(b_id) && 
              ( (!inverted(f_id) && constr->content_rank >= max_rank) 
               || (inverted(f_id) && constr->content_rank <= min_rank)) )
         { int methnum;
           methnum = create_body_constraint_content_method(b_id,n);
           attach_method_num(get_body_volquant(b_id),methnum);
           apply_method_num(v_id,methnum);
         }
       } while ( !equal_id(fe,first_fe) );
       e_id = get_next_tail_edge(e_id);
     } while ( !equal_id(first_e,e_id));
   }
  }
} // end set_v_constraint_map()

/**************************************************************************
*
* Function: unset_v_constraint_map(
*
* Purpose: Remove an individual level-set constraint from a vertex.
*/
void unset_v_constraint_map(    
  vertex_id v_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;
  int k,j;

  n &= CONMASK;
  if ( maxcon == 0 ) return;
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
     if ( (map[k] & CONMASK) == (conmap_t)n ) break;
  if ( k > (int)*map ) return;
  map[0]--;
  for ( j = k ; j <= (int)*map ; j++ ) map[j] = map[j+1];
  map[j] = 0;
  if ( map[0] == 0 ) unset_attr(v_id,CONSTRAINT);
  if ( everything_quantities_flag )
  { // delete energy and content methods
    int i;
    int meth_offset = get_meth_offset(VERTEX);
    struct vertex *vptr = (struct vertex *)elptr(v_id);
    int *instlist = (int*)((char*)vptr + meth_offset);
    for ( i = 0 ; i < vptr->method_count ; i++ )
    { struct method_instance *mi = METH_INSTANCE(abs(instlist[i]));
      if ( mi->connum == n )
      { instlist[i] = instlist[--vptr->method_count];
        i--;
      }
    }
  }
} // end unset_v_constraint_map()

/**************************************************************************
*
* Function: unset_v_all_constraints()
*
* Purpose: Remove all level-set constraints from a vertex.
*/
void unset_v_all_constraints(vertex_id v_id)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;
  int i;

  if ( maxcon == 0 ) return;
  map = get_v_constraint_map(v_id);

  for ( i = map[0] ; i >= 1 ; i-- )
    unset_v_constraint_map(v_id,map[i]);

  unset_attr(v_id,CONSTRAINT);

} // end unset_v_all_constraints()

/**************************************************************************
*
* Function: v_on_constraint()
*
* Purpose: Determine whether a vertex is on a given level-set constraint.
*/
int v_on_constraint( 
  vertex_id v_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;
  int k;

  n &= CONMASK;
  if ( maxcon == 0 ) return 0;
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { if ( (map[k] & CONMASK) == (conmap_t)n ) 
        return 1;
  }
  return 0;
} // end v_on_constraint()

/**************************************************************************
*
* Function: v_hit_constraint_count()
*
* Purpose: Determine how many level-set constraints a vertex has hit.
*/
int v_hit_constraint_count(vertex_id v_id)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;
  int count = 0;
  int k;

  if ( maxcon == 0 ) return 0;
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { if ( (map[k] & CON_HIT_BIT) ) 
      count++;
  }
  return count;
} // end v_hit_constraint_count()

/**************************************************************************
*
* Function: get_v_common_conmap()
*
* Purpose: Determine the common level-set constraints of two vertices.
*/
void get_v_common_conmap(
  vertex_id v1,
  vertex_id v2,
  conmap_t *conmap, // for return list
  int max   // size of conmap
)
{ conmap_t *map1 = get_v_constraint_map(v1);
  unsigned int k;

  conmap[0] = 0;
  for ( k = 1; k <= map1[0] ; k++ )
    if ( v_on_constraint(v2,map1[k]) )
    { if ( conmap[0] >= (unsigned)(max-1) )
      { sprintf(errmsg,"Too many common constraints for vertices %s and %s.\n",
          ELNAME(v1),ELNAME2(v2));
        kb_error(5891,errmsg,RECOVERABLE);
      }
      conmap[++conmap[0]] = map1[k] & (conmap_t)CONMASK;
    }
} // end get_v_common_conmap()

/**************************************************************************
*
* Function: get_v_constraint_status()
*
* Purpose: See if a vertex has hit a given level-set constraint.
*/
int get_v_constraint_status(v_id,n)  
vertex_id v_id;
int n;
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;
  int k;

  n &= CONMASK;
  if ( maxcon == 0 ) return 0;
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { if ( (map[k] & CONMASK) == (conmap_t)n ) 
        return (map[k] & CON_HIT_BIT) ? 1 : 0;
  }
  return 0;
} // end get_v_constraint_status()

/**************************************************************************
*
* Function: clear_v_conmap()
*
* Purpose: Empty a vertex's level-set constraint list.
*/
void clear_v_conmap(vertex_id v_id)
{ conmap_t *map = get_v_constraint_map(v_id);
  unsigned int k;

  for ( k = 1 ; k <= map[0] ; k++ )
    map[k] = 0;
  map[0] = 0;
} // end clear_v_conmap()

/**************************************************************************
*
* Function: set_v_constraint_status()
*
* Purpose: Mark a vertex as hitting a given level-set constraint.
*/
void set_v_constraint_status(  
  vertex_id v_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;

  int k;
  n &= CONMASK;
  if ( maxcon == 0 ) return;
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { if ( (map[k] & CONMASK) == (conmap_t)n ) 
    { map[k] |= CON_HIT_BIT;
      set_attr(v_id,HIT_WALL);
      return;
    }
  }
} // end set_v_constraint_status()

/**************************************************************************
*
* Function: unset_v_constraint_status()
*
* Purpose: Mark a vertex as not hitting a given level-set constraint.
*/
void unset_v_constraint_status(
  vertex_id v_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;

  int k;
  n &= CONMASK;
  if ( maxcon == 0 ) return;
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { if ( (map[k] & CONMASK) == (conmap_t)n ) 
     { map[k] &= ~CON_HIT_BIT;
       return;
     }
  }
} // end unset_v_constraint_status()

/**************************************************************************
*
* Function: clear_v_constraint_status()
*
* Purpose: Mark vertex as hitting no one-sided level-set constraints at all.
*/
void clear_v_constraint_status(vertex_id v_id)
{ conmap_t *map;
  int maxcon = EXTRAS(VERTEX)[V_CONSTR_LIST_ATTR].array_spec.datacount;
  int k;

  if ( maxcon == 0 ) return;
  map = get_v_constraint_map(v_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { struct constraint *con = get_constraint(map[k]);
    if ( con->attr & (NONPOSITIVE|NONNEGATIVE) )
      map[k] &= ~CON_HIT_BIT;
  }

} // end clear_v_constraint_status()
 
/**************************************************************************
*
* Function: assure_v_constraints()
*
* Purpose: Guarantee proper vertex constraint-associated quantities
*          by removing and resetting all level-set constraints.
*/
void assure_v_constraints(vertex_id v_id)
{ conmap_t *map;
  conmap_t tempmap[MAXCONPER],*tmp=NULL;
  int k;
  unsigned int kk;

  map = get_v_constraint_map(v_id);
  if ( map[0] > MAXCONPER )
    tmp = (conmap_t *)temp_calloc(map[0]+5,sizeof(conmap_t));
  for ( k = 0; k <= (int)*map ; k++ )
    tempmap[k] = map[k];
  unset_v_all_constraints(v_id);
  for ( kk = 1; kk <= tempmap[0] ; kk++ )
    set_v_constraint_map(v_id,tempmap[kk]);
  if ( tmp )
    temp_free((char*)tmp);

} // end assure_v_constraints()

/**************************************************************************
*
* Function: set_e_constraint_map()
*
* Purpose: Put an edge on a given level-set constraint.
*/
void set_e_constraint_map(  
  edge_id e_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(EDGE)[E_CONSTR_LIST_ATTR].array_spec.datacount;
  int k;
  struct constraint *constr;
  int four = 4;

  n &= CONMASK;
  if ( maxcon == 0 )
     expand_attribute(EDGE,E_CONSTR_LIST_ATTR,&four);
  map = get_e_constraint_map(e_id);
  for ( k = 1; k <= (int)*map ; k++ )
     if ( (map[k] & CONMASK) == (conmap_t)n ) return;
  if ( k >= maxcon )
  { int newmax;
    newmax = maxcon+4;
    expand_attribute(EDGE,E_CONSTR_LIST_ATTR,&newmax);
    map = get_e_constraint_map(e_id);
  } 
  map[k] = (conmap_t)n; 
  map[0]++; /* counter */

  set_attr(e_id,CONSTRAINT);
  constr = get_constraint(n);
  if ( constr->attr & CON_ENERGY )
  { set_attr(e_id, BDRY_ENERGY);
    if ( everything_quantities_flag )
       apply_method_num(positive_id(e_id),constr->energy_method);
/* positive_id() to agree with old way */
  }
  if ( constr->attr & CON_CONTENT )
  { set_attr(e_id, BDRY_CONTENT);  /* BIG PROBLEM HERE GETTING RIGHT BODY!!! */
    if ( everything_quantities_flag )
    { facetedge_id fe,first_fe = get_edge_fe(e_id);
      int min_rank, max_rank,j;

      min_rank = MAXINT; max_rank = 0;
      for ( j = 1 ; j <= (int)map[0] ; j++ )
      { 
        struct constraint *c = get_constraint(map[j]);
        if ( c->content_rank < min_rank ) min_rank = c->content_rank;
        if ( c->content_rank > max_rank ) max_rank = c->content_rank;
      }
      fe = first_fe;
      if ( valid_id(first_fe) )
      do
      { 
        body_id b_id;
        facet_id f_id;
        f_id = get_fe_facet(fe);
        fe = get_next_facet(fe);
        if ( !valid_id(f_id) ) continue;
        if ( get_fattr(f_id) & NONCONTENT ) continue;
        b_id = get_facet_body(f_id);
        if ( valid_id(b_id) && (constr->content_rank >= max_rank))
        { int methnum;
          methnum = create_body_constraint_content_method(b_id,n);
          attach_method_num(get_body_volquant(b_id),methnum);
          apply_method_num(e_id,methnum);
        }
        b_id = get_facet_body(inverse_id(f_id));
        if ( valid_id(b_id) && (constr->content_rank <= min_rank))
        { int methnum;
          methnum = create_body_constraint_content_method(b_id,n);
          attach_method_num(get_body_volquant(b_id),methnum);
          apply_method_num(inverse_id(e_id),methnum);
        }
      } while ( !equal_id(fe,first_fe) );

    }
  }
} // end set_e_constraint_map()

/**************************************************************************
*
* Function: e_on_constraint()
*
* Purpose: Determine if a vertex is on a given level-set constraint.
*/
int e_on_constraint(e_id,n)  
edge_id e_id;
int n;
{ conmap_t *map;
  int maxcon = EXTRAS(EDGE)[E_CONSTR_LIST_ATTR].array_spec.datacount;
  int k;

  n &= CONMASK;
  if ( maxcon == 0 ) return 0;
  map = get_e_constraint_map(e_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { if ( (map[k] & CONMASK) == (conmap_t)n ) 
        return 1;
  }
  return 0;
} // end e_on_constraint()

/**************************************************************************
*
* Function: unset_e_constraint_map()
*
* Purpose: Remove an edge from a level-set constraint.
*/
void unset_e_constraint_map(
  edge_id e_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(EDGE)[E_CONSTR_LIST_ATTR].array_spec.datacount;
  int j,k;

  n &= CONMASK;
  if ( maxcon == 0 ) return;
  map = get_e_constraint_map(e_id);
  for ( k = 1; k <= (int)*map ; k++ )
    if ( (map[k] & CONMASK) == (conmap_t)n ) break;
  if ( k > (int)*map ) return;

  map[0]--;
  for ( j = k ; j <= (int)*map ; j++ ) map[j] = map[j+1];
  map[j] = 0;
  if ( map[0] == 0 ) unset_attr(e_id,CONSTRAINT);

  if ( everything_quantities_flag )
  { 
    // doing low-level deletion here since may be many volume methods
    // associated with constraint, too many to list in constraint structure.
    int meth_offset = get_meth_offset(EDGE);
    int i;
    struct element *eptr = elptr(e_id);
    int *instlist = (int*)((char*)eptr + meth_offset);
    for ( i = 0 ; i < eptr->method_count ; i++ )
    { struct method_instance *mi;
      mi = METH_INSTANCE(abs(instlist[i]));
      if ( mi->connum == n)
      { instlist[i] = instlist[--eptr->method_count];
        i--;
      }
    }
  }

} // end unset_e_constraint_map()

/**************************************************************************
*
* Function: unset_e_all_constraints()
*
* Purpose: Remove an edge from all level-set constraints.
*/
void unset_e_all_constraints(edge_id e_id)
{ conmap_t *map;
  int maxcon = EXTRAS(EDGE)[E_CONSTR_LIST_ATTR].array_spec.datacount;
  int i;

  if ( maxcon == 0 ) return;
  map = get_e_constraint_map(e_id);

  for ( i = map[0] ; i >= 1 ; i-- )
    unset_e_constraint_map(e_id,map[i]);
} // end unset_e_all_constraints()

/**************************************************************************
*
* Function: set_f_constraint_map()
*
* Purpose: Set a facet on a level-set constraint.
*/
void set_f_constraint_map(  
  facet_id f_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(FACET)[F_CONSTR_LIST_ATTR].array_spec.datacount;
  int k;
  int four = 4;

  n &= CONMASK;  /* get rid of hit bit */
  if ( maxcon == 0 )
     expand_attribute(FACET,F_CONSTR_LIST_ATTR,&four);
  map = get_f_constraint_map(f_id);
  for ( k = 1; k <= (int)*map ; k++ )
     if ( (map[k] & CONMASK) == (conmap_t)n ) return;
  if ( k >= maxcon )
  { int newmax;
    newmax = maxcon+4;
    expand_attribute(FACET,F_CONSTR_LIST_ATTR,&newmax);
    map = get_f_constraint_map(f_id);
  } 
  map[k] = (conmap_t)n; 
  map[0]++; /* counter */
} // end set_f_constraint_map()

/**************************************************************************
*
* Function: f_on_constraint()
*
* Purpose: Determine if a facet is on a given level-set constraint.
*/
int f_on_constraint(  
  facet_id f_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(FACET)[F_CONSTR_LIST_ATTR].array_spec.datacount;
  int k;

  n &= CONMASK;
  if ( maxcon == 0 ) return 0;
  map = get_f_constraint_map(f_id);
  for ( k = 1; k <= (int)*map ; k++ )
  { if ( (map[k] & CONMASK) == (conmap_t)n ) 
        return 1;
  }
  return 0;
} // end  f_on_constraint()

/**************************************************************************
*
* Function: unset_f_constraint_map()
*
* Purpose: Remove a facet from a level-set constraint.
*/
void unset_f_constraint_map(    
  facet_id f_id,
  int n
)
{ conmap_t *map;
  int maxcon = EXTRAS(FACET)[F_CONSTR_LIST_ATTR].array_spec.datacount;
  int k,j;

  n &= CONMASK;
  if ( maxcon == 0 ) return;
  map = get_f_constraint_map(f_id);
  for ( k = 1; k <= (int)*map ; k++ )
     if ( (map[k] & CONMASK) == (conmap_t)n ) break;
  if ( k > (int)*map ) return;
  map[0]--;
  for ( j = k ; j <= (int)*map ; j++ ) map[j] = map[j+1];
  map[j] = 0;
  if ( map[0] == 0 ) unset_attr(f_id,CONSTRAINT);
  if ( everything_quantities_flag )
  { struct constraint *con = get_constraint(n);
    if ( con->attr & CON_ENERGY )
       unapply_method(f_id,con->energy_method);
  }
} // end unset_f_constraint_map()

/**************************************************************************
*
* Function: unset_f_all_constraints()
*
* Purpose: Remove a facet from all level-set constraints.
*/
void unset_f_all_constraints(facet_id f_id)
{ conmap_t *map;
  int maxcon = EXTRAS(FACET)[F_CONSTR_LIST_ATTR].array_spec.datacount;
  int i;

  if ( maxcon == 0 ) return;
  map = get_f_constraint_map(f_id);

  for ( i = map[0] ; i >= 1 ; i-- )
    unset_f_constraint_map(f_id,map[i]);
} // end unset_f_all_constraints()

