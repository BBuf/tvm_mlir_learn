#ifndef _SMTL_H
#define _SMTL_H

typedef struct smtl_t* smtl_handle;
typedef void (*task_func_t)(void*);

void smtl_init(smtl_handle *psh,
    int num_threads);

void smtl_fini(smtl_handle sh);

void smtl_add_task(smtl_handle sh,
    task_func_t task_func,
    void *params);

void smtl_begin_tasks(smtl_handle sh);

void smtl_wait_tasks_finished(smtl_handle sh);

#endif

