#define _GNU_SOURCE

#include "smtl.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>

#define SMTL_MAX_THREADS 128

enum smtl_status
{
    SMTL_WORK,
    SMTL_IDLE,
    SMTL_FINI,
};

struct queue_node_t
{
    task_func_t task_func;
    void *params;
    struct queue_node_t *next;
};

struct smtl_t
{
    int num_threads;

    struct queue_node_t *task_queues[SMTL_MAX_THREADS];
    int cur_qid;

    pthread_t tids[SMTL_MAX_THREADS];

    pthread_mutex_t pt_mtx;
    pthread_cond_t pt_cv;
    int thread_holds;

    pthread_mutex_t sl_mtxs[SMTL_MAX_THREADS];
    pthread_cond_t sl_cvs[SMTL_MAX_THREADS];
    enum smtl_status status[SMTL_MAX_THREADS];
};

struct smtl_tp_t
{
    int tid;
    struct smtl_t *sh;
};

static void thread_bind(int cpu)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    if (pthread_setaffinity_np(pthread_self(),
            sizeof(cpu_set_t), &cpu_set) != 0)
    {
        fprintf(stderr, "Error: cpu[%d] bind failed.\n", cpu);
        exit(0);
    }
}

static void *smtl_thread_func(void *params)
{
    int err = 0;

    struct smtl_tp_t *stp = (struct smtl_tp_t*)params;
    int tid = stp->tid;
    struct smtl_t *sh = stp->sh;
    free(stp);

    thread_bind(tid);

    pthread_mutex_t *sl_mtx = sh->sl_mtxs + tid;
    pthread_cond_t *sl_cv = sh->sl_cvs + tid;

    pthread_mutex_t *pt_mtx = &sh->pt_mtx;
    pthread_cond_t *pt_cv = &sh->pt_cv;

    while (1)
    {
        err = pthread_mutex_lock(sl_mtx);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtx lock failed.\n");
            exit(0);
        }
        while (sh->status[tid] != SMTL_WORK)
        {
            if (sh->status[tid] == SMTL_FINI)
            {
                err = pthread_mutex_unlock(sl_mtx);
                if (err != 0)
                {
                    fprintf(stderr, "ERROR: sl_mtx unlock failed.\n");
                    exit(0);
                }
                return NULL;
            }
            err = pthread_cond_wait(sl_cv, sl_mtx);
            if (err != 0)
            {
                fprintf(stderr, "ERROR: sl_cv wait failed.\n");
                exit(0);
            }
        }
        err = pthread_mutex_unlock(sl_mtx);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtx unlock failed.\n");
            exit(0);
        }

        struct queue_node_t *p = sh->task_queues[tid];
        struct queue_node_t *q = NULL;
        sh->task_queues[tid] = NULL;
        while (p != NULL)
        {
            q = p->next;
            p->task_func(p->params);
            free(p);
            p = q;
        }

        err = pthread_mutex_lock(pt_mtx);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: pt_mtx lock failed.\n");
            exit(0);
        }
        sh->status[tid] = SMTL_IDLE;
        sh->thread_holds--;
        if (sh->thread_holds == 0)
        {
            err = pthread_cond_signal(pt_cv);
            if (err != 0)
            {
                fprintf(stderr, "ERROR: pt_cv signal failed.\n");
                exit(0);
            }
        }
        err = pthread_mutex_unlock(pt_mtx);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: pt_mtx unlock failed.\n");
            exit(0);
        }
    }

    return NULL;
}

void smtl_init(smtl_handle *psh,
    int num_threads)
{
    int err = 0;

    struct smtl_t *sh =
        (struct smtl_t*)malloc(sizeof(struct smtl_t));
    if (sh == NULL)
    {
        fprintf(stderr,
            "ERROR: smtl_init allocation failed.\n");
        exit(0);
    }

    sh->num_threads = num_threads;
    sh->cur_qid = 0;
    sh->thread_holds = 0;

    memset(sh->task_queues, 0,
        num_threads * sizeof(struct queue_node_t*));

    err = pthread_mutex_init(&sh->pt_mtx, NULL);
    if (err != 0)
    {
        fprintf(stderr, "ERROR: pt_mtx init failed.\n");
        exit(0);
    }

    err = pthread_cond_init(&sh->pt_cv, NULL);
    if (err != 0)
    {
        fprintf(stderr, "ERROR: pt_cv init failed.\n");
        exit(0);
    }

    int i;
    for (i = 0; i < num_threads; i++)
    {
        err = pthread_mutex_init(sh->sl_mtxs + i, NULL);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtxs init failed.\n");
            exit(0);
        }

        err = pthread_cond_init(sh->sl_cvs + i, NULL);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_cvs init failed.\n");
            exit(0);
        }

        sh->status[i] = SMTL_IDLE;

        struct smtl_tp_t *stp =
            (struct smtl_tp_t*)malloc(sizeof(struct smtl_tp_t));
        if (stp == NULL)
        {
            fprintf(stderr, "ERROR: stp allocation failed.\n");
            exit(0);
        }
        stp->sh = sh;
        stp->tid = i;

        err = pthread_create(sh->tids + i, NULL,
            smtl_thread_func, stp);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: pthread_create failed.\n");
            exit(0);
        }
    }
    *psh = sh;
}

void smtl_fini(smtl_handle sh)
{
    int err = 0;
    int i;
    for (i = 0; i < sh->num_threads; i++)
    {
        err = pthread_mutex_lock(sh->sl_mtxs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtxs lock failed.\n");
            exit(0);
        }
        sh->status[i] = SMTL_FINI;
        err = pthread_cond_signal(sh->sl_cvs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_cv signal failed.\n");
            exit(0);
        }
        err = pthread_mutex_unlock(sh->sl_mtxs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtxs unlock failed.\n");
            exit(0);
        }
    }

    for (i = 0; i < sh->num_threads; i++)
    {
        err = pthread_join(sh->tids[i], NULL);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: pthread_join failed.\n");
            exit(0);
        }
    }

    err = pthread_mutex_destroy(&sh->pt_mtx);
    if (err != 0)
    {
        fprintf(stderr, "ERROR: pt_mtx destroy failed.\n");
        exit(0);
    }
    err = pthread_cond_destroy(&sh->pt_cv);
    if (err != 0)
    {
        fprintf(stderr, "ERROR: pt_cv destroy failed.\n");
        exit(0);
    }
    for (i = 0; i < sh->num_threads; i++)
    {
        err = pthread_mutex_destroy(sh->sl_mtxs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtxs destroy failed.\n");
            exit(0);
        }

        err = pthread_cond_destroy(sh->sl_cvs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_cvs destroy failed.\n");
            exit(0);
        }

        struct queue_node_t *p = sh->task_queues[i], *q = NULL;
        while (p != NULL)
        {
            q = p->next;
            free(p);
            p = q;
        }
    }
}

void smtl_add_task(smtl_handle sh,
    task_func_t task_func,
    void *params)
{
    struct queue_node_t *task =
        (struct queue_node_t*)malloc(sizeof(struct queue_node_t));
    if (task == NULL)
    {
        fprintf(stderr, "ERROR: add_task allocation failed.\n");
        exit(0);
    }

    task->task_func = task_func;
    task->params = params;
    task->next = sh->task_queues[sh->cur_qid];
    sh->task_queues[sh->cur_qid] = task;
    sh->cur_qid++;
    if (sh->cur_qid == sh->num_threads)
    {
        sh->cur_qid = 0;
    }
}

void smtl_begin_tasks(smtl_handle sh)
{
    int i, err = 0;
    sh->thread_holds = sh->num_threads;
    for (i = 0; i < sh->num_threads; i++)
    {
        err = pthread_mutex_lock(sh->sl_mtxs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtxs lock failed.\n");
            exit(0);
        }
        sh->status[i] = SMTL_WORK;
        err = pthread_cond_signal(sh->sl_cvs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_cvs signal failed.\n");
            exit(0);
        }
        err = pthread_mutex_unlock(sh->sl_mtxs + i);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: sl_mtxs unlock failed.\n");
            exit(0);
        }
    }
}

void smtl_wait_tasks_finished(smtl_handle sh)
{
    int err = 0;

    pthread_mutex_lock(&sh->pt_mtx);
    if (err != 0)
    {
        fprintf(stderr, "ERROR: pt_mtx lock failed.\n");
        exit(0);
    }
    while (sh->thread_holds > 0)
    {
        pthread_cond_wait(&sh->pt_cv, &sh->pt_mtx);
        if (err != 0)
        {
            fprintf(stderr, "ERROR: pt_cv wait failed.\n");
            exit(0);
        }
    }
    sh->cur_qid = 0;
    pthread_mutex_unlock(&sh->pt_mtx);
    if (err != 0)
    {
        fprintf(stderr, "ERROR: pt_mtx unlock failed.\n");
        exit(0);
    }
}

