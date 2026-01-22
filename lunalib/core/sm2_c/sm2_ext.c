#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include "sm2_bn.h"
#include "sm2_ec.h"

typedef struct {
    PyObject* sm2_instance;
    PyObject* priv;
    PyObject* d;
    PyObject* inv;
} Sm2ThreadCache;

static Py_tss_t sm2_cache_key = Py_tss_NEEDS_INIT;

static Sm2ThreadCache* sm2_get_cache(void) {
    Sm2ThreadCache* cache = (Sm2ThreadCache*)PyThread_tss_get(&sm2_cache_key);
    if (!cache) {
        cache = (Sm2ThreadCache*)PyMem_Calloc(1, sizeof(Sm2ThreadCache));
        if (!cache) {
            return NULL;
        }
        PyThread_tss_set(&sm2_cache_key, (void*)cache);
    }
    return cache;
}

static int sm2_debug_enabled(void) {
    PyObject* env = PySys_GetObject("_sm2_ext_debug");
    if (env && PyUnicode_Check(env)) {
        return PyUnicode_GetLength(env) > 0;
    }
    PyObject* os_mod = PyImport_ImportModule("os");
    if (!os_mod) {
        return 0;
    }
    PyObject* getenv = PyObject_GetAttrString(os_mod, "getenv");
    Py_DECREF(os_mod);
    if (!getenv) {
        return 0;
    }
    PyObject* val = PyObject_CallFunction(getenv, "s", "SM2_EXT_DEBUG");
    Py_DECREF(getenv);
    if (!val) {
        return 0;
    }
    int enabled = PyUnicode_Check(val) && PyUnicode_GetLength(val) > 0;
    Py_DECREF(val);
    if (enabled) {
        PySys_SetObject("_sm2_ext_debug", PyUnicode_FromString("1"));
    }
    return enabled;
}

static PyObject* sm2_sign(PyObject* self, PyObject* args) {
    if (sm2_debug_enabled()) {
        PySys_WriteStderr("[sm2_ext] sign: begin\n");
    }
    Py_buffer msg;
    Py_buffer priv;
    if (!PyArg_ParseTuple(args, "y*y*", &msg, &priv)) {
        return NULL;
    }
    PyObject* sm2_instance = NULL;
    Sm2ThreadCache* cache = sm2_get_cache();
    if (cache && cache->sm2_instance) {
        sm2_instance = cache->sm2_instance;
        Py_INCREF(sm2_instance);
    } else {
        PyObject* sm2_module = PyImport_ImportModule("lunalib.core.sm2");
        if (!sm2_module) {
            PyBuffer_Release(&msg);
            PyBuffer_Release(&priv);
            return NULL;
        }
        PyObject* sm2_class = PyObject_GetAttrString(sm2_module, "SM2");
        Py_DECREF(sm2_module);
        if (!sm2_class) {
            PyBuffer_Release(&msg);
            PyBuffer_Release(&priv);
            return NULL;
        }
        sm2_instance = PyObject_CallObject(sm2_class, NULL);
        Py_DECREF(sm2_class);
        if (!sm2_instance) {
            PyBuffer_Release(&msg);
            PyBuffer_Release(&priv);
            return NULL;
        }
        if (cache) {
            cache->sm2_instance = sm2_instance;
            Py_INCREF(cache->sm2_instance);
        }
    }

    // Get Z and curve.n
    PyObject* z_val = PyObject_GetAttrString(sm2_instance, "Z");
    PyObject* curve = PyObject_GetAttrString(sm2_instance, "curve");
    PyObject* n_val = curve ? PyObject_GetAttrString(curve, "n") : NULL;
    Py_XDECREF(curve);
    if (!z_val || !n_val) {
        Py_XDECREF(z_val);
        Py_XDECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }

    // e = H(Z || msg)
    if (!PyBytes_Check(z_val)) {
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* msg_bytes = PyBytes_FromStringAndSize((const char*)msg.buf, msg.len);
    if (!msg_bytes) {
        Py_DECREF(z_val);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    Py_ssize_t z_len = PyBytes_GET_SIZE(z_val);
    PyObject* zmsg = PyBytes_FromStringAndSize(NULL, z_len + msg.len);
    if (!zmsg) {
        Py_DECREF(msg_bytes);
        Py_DECREF(z_val);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    char* zmsg_buf = PyBytes_AS_STRING(zmsg);
    memcpy(zmsg_buf, PyBytes_AS_STRING(z_val), z_len);
    memcpy(zmsg_buf + z_len, PyBytes_AS_STRING(msg_bytes), msg.len);
    Py_DECREF(msg_bytes);
    Py_DECREF(z_val);
    if (!zmsg) {
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* hash_obj = PyObject_GetAttrString(sm2_instance, "hash");
    PyObject* hash_fn = hash_obj ? PyObject_GetAttrString(hash_obj, "hash") : NULL;
    Py_XDECREF(hash_obj);
    if (!hash_fn) {
        Py_DECREF(zmsg);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* e_bytes = PyObject_CallFunctionObjArgs(hash_fn, zmsg, NULL);
    Py_DECREF(hash_fn);
    Py_DECREF(zmsg);
    if (!e_bytes) {
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    if (sm2_debug_enabled()) {
        PySys_WriteStderr("[sm2_ext] sign: hash done\n");
    }
    PyObject* e_int = PyObject_CallMethod((PyObject*)&PyLong_Type, "from_bytes", "Os", e_bytes, "big");
    if (!e_int) {
        Py_DECREF(e_bytes);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }

    // d from priv (with cache)
    PyObject* priv_bytes = PyBytes_FromStringAndSize((const char*)priv.buf, priv.len);
    if (!priv_bytes) {
        Py_DECREF(e_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* d_int = NULL;
    PyObject* inv_int = NULL;
    if (cache && cache->priv && cache->d && cache->inv) {
        int eq = PyObject_RichCompareBool(cache->priv, priv_bytes, Py_EQ);
        if (eq == 1) {
            d_int = cache->d;
            inv_int = cache->inv;
            Py_INCREF(d_int);
            Py_INCREF(inv_int);
            if (sm2_debug_enabled()) {
                PySys_WriteStderr("[sm2_ext] sign: cache hit\n");
            }
        } else if (eq < 0) {
            Py_DECREF(priv_bytes);
            Py_DECREF(e_int);
            Py_DECREF(n_val);
            Py_DECREF(sm2_instance);
            PyBuffer_Release(&msg);
            PyBuffer_Release(&priv);
            return NULL;
        }
    }
    if (!d_int) {
        d_int = PyObject_CallMethod((PyObject*)&PyLong_Type, "from_bytes", "Os", priv_bytes, "big");
        if (!d_int) {
            Py_DECREF(priv_bytes);
            Py_DECREF(e_int);
            Py_DECREF(n_val);
            Py_DECREF(sm2_instance);
            PyBuffer_Release(&msg);
            PyBuffer_Release(&priv);
            return NULL;
        }
    }
    Py_DECREF(priv_bytes);

    // k = secrets.randbelow(n-1) + 1
    PyObject* secrets_mod = PyImport_ImportModule("secrets");
    PyObject* randbelow = secrets_mod ? PyObject_GetAttrString(secrets_mod, "randbelow") : NULL;
    Py_XDECREF(secrets_mod);
    if (!randbelow) {
        Py_DECREF(d_int);
        Py_DECREF(e_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* n_minus_1 = PyNumber_Subtract(n_val, PyLong_FromLong(1));
    PyObject* k_int = PyObject_CallFunctionObjArgs(randbelow, n_minus_1, NULL);
    Py_DECREF(randbelow);
    Py_DECREF(n_minus_1);
    if (!k_int) {
        Py_DECREF(d_int);
        Py_DECREF(e_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* one = PyLong_FromLong(1);
    PyObject* k_plus_1 = PyNumber_Add(k_int, one);
    Py_DECREF(one);
    Py_DECREF(k_int);
    if (!k_plus_1) {
        Py_DECREF(d_int);
        Py_DECREF(e_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }

    // k*G using C
    PyObject* k_bytes = PyObject_CallMethod(k_plus_1, "to_bytes", "is", 32, "big");
    if (!k_bytes) {
        Py_DECREF(k_plus_1);
        Py_DECREF(d_int);
        Py_DECREF(e_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    sm2_bn256 k_bn;
    sm2_bn_from_bytes_be(&k_bn, (const uint8_t*)PyBytes_AsString(k_bytes));
    Py_DECREF(k_bytes);

    sm2_point G;
    sm2_point R;
    sm2_point_set_generator(&G);
    sm2_point_scalar_mul(&R, &G, &k_bn);
    if (sm2_debug_enabled()) {
        PySys_WriteStderr("[sm2_ext] sign: scalar mul done\n");
    }

    uint8_t x1_bytes[32];
    sm2_bn_to_bytes_be(&R.x, x1_bytes);
    // r = (e + x1) mod n (computed in C)
    sm2_bn256 e_bn;
    sm2_bn256 x1_bn;
    sm2_bn256 r_bn;
    sm2_bn_from_bytes_be(&e_bn, (const uint8_t*)PyBytes_AsString(e_bytes));
    sm2_bn_from_bytes_be(&x1_bn, (const uint8_t*)x1_bytes);
    sm2_bn_mod_n_add(&r_bn, &e_bn, &x1_bn);
    uint8_t r_bytes_buf[32];
    sm2_bn_to_bytes_be(&r_bn, r_bytes_buf);
    PyObject* r_bytes_local = PyBytes_FromStringAndSize((const char*)r_bytes_buf, 32);
    PyObject* r_int = PyObject_CallMethod((PyObject*)&PyLong_Type, "from_bytes", "Os", r_bytes_local, "big");
    Py_DECREF(r_bytes_local);
    Py_DECREF(e_int);
    Py_DECREF(e_bytes);
    if (!r_int) {
        Py_DECREF(k_plus_1);
        Py_DECREF(d_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }

    if (sm2_debug_enabled()) {
        PySys_WriteStderr("[sm2_ext] sign: r computed\n");
    }

    // s = inv(1+d) * (k - r*d) mod n (Python pow to avoid C inversion hang)
    if (!inv_int) {
        PyObject* one_py = PyLong_FromLong(1);
        PyObject* d_plus1_int = PyNumber_Add(d_int, one_py);
        Py_DECREF(one_py);
        if (!d_plus1_int) {
            Py_DECREF(k_plus_1);
            Py_DECREF(r_int);
            Py_DECREF(d_int);
            Py_DECREF(n_val);
            Py_DECREF(sm2_instance);
            PyBuffer_Release(&msg);
            PyBuffer_Release(&priv);
            return NULL;
        }
        if (sm2_debug_enabled()) {
            PySys_WriteStderr("[sm2_ext] sign: inv start\n");
        }
        PyObject* exp_neg_one = PyLong_FromLong(-1);
        inv_int = PyNumber_Power(d_plus1_int, exp_neg_one, n_val);
        Py_DECREF(exp_neg_one);
        Py_DECREF(d_plus1_int);
        if (!inv_int) {
            Py_DECREF(k_plus_1);
            Py_DECREF(r_int);
            Py_DECREF(d_int);
            Py_DECREF(n_val);
            Py_DECREF(sm2_instance);
            PyBuffer_Release(&msg);
            PyBuffer_Release(&priv);
            return NULL;
        }
        if (sm2_debug_enabled()) {
            PySys_WriteStderr("[sm2_ext] sign: inv done\n");
        }
        if (cache) {
            if (cache->priv) {
                Py_DECREF(cache->priv);
            }
            cache->priv = PyBytes_FromStringAndSize((const char*)priv.buf, priv.len);
            if (cache->d) {
                Py_DECREF(cache->d);
            }
            if (cache->inv) {
                Py_DECREF(cache->inv);
            }
            cache->d = d_int;
            cache->inv = inv_int;
            Py_INCREF(cache->d);
            Py_INCREF(cache->inv);
        }
    }
    PyObject* rd_int = PyNumber_Multiply(r_int, d_int);
    PyObject* k_minus_rd_int = rd_int ? PyNumber_Subtract(k_plus_1, rd_int) : NULL;
    Py_XDECREF(rd_int);
    if (!k_minus_rd_int) {
        Py_DECREF(inv_int);
        Py_DECREF(k_plus_1);
        Py_DECREF(r_int);
        Py_DECREF(d_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* k_minus_rd_mod = PyNumber_Remainder(k_minus_rd_int, n_val);
    Py_DECREF(k_minus_rd_int);
    if (!k_minus_rd_mod) {
        Py_DECREF(inv_int);
        Py_DECREF(k_plus_1);
        Py_DECREF(r_int);
        Py_DECREF(d_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* s_tmp = PyNumber_Multiply(inv_int, k_minus_rd_mod);
    Py_DECREF(inv_int);
    Py_DECREF(k_minus_rd_mod);
    if (!s_tmp) {
        Py_DECREF(k_plus_1);
        Py_DECREF(r_int);
        Py_DECREF(d_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    PyObject* s_mod = PyNumber_Remainder(s_tmp, n_val);
    Py_DECREF(s_tmp);
    if (!s_mod) {
        Py_DECREF(k_plus_1);
        Py_DECREF(r_int);
        Py_DECREF(d_int);
        Py_DECREF(n_val);
        Py_DECREF(sm2_instance);
        PyBuffer_Release(&msg);
        PyBuffer_Release(&priv);
        return NULL;
    }
    if (sm2_debug_enabled()) {
        PySys_WriteStderr("[sm2_ext] sign: s computed\n");
    }

    Py_DECREF(k_plus_1);
    Py_DECREF(n_val);
    Py_DECREF(sm2_instance);
    PyBuffer_Release(&msg);
    PyBuffer_Release(&priv);

    PyObject* s_int = s_mod;
    if (!s_int) {
        Py_DECREF(r_int);
        return NULL;
    }

    PyObject* r_bytes = PyObject_CallMethod(r_int, "to_bytes", "is", 32, "big");
    PyObject* s_bytes = PyObject_CallMethod(s_int, "to_bytes", "is", 32, "big");
    Py_DECREF(r_int);
    Py_DECREF(s_int);
    if (!r_bytes || !s_bytes) {
        Py_XDECREF(r_bytes);
        Py_XDECREF(s_bytes);
        return NULL;
    }
    PyObject* sig = PyBytes_FromStringAndSize(NULL, 64);
    if (!sig) {
        Py_DECREF(r_bytes);
        Py_DECREF(s_bytes);
        return NULL;
    }
    char* sig_buf = PyBytes_AS_STRING(sig);
    memcpy(sig_buf, PyBytes_AS_STRING(r_bytes), 32);
    memcpy(sig_buf + 32, PyBytes_AS_STRING(s_bytes), 32);
    Py_DECREF(r_bytes);
    Py_DECREF(s_bytes);
    if (sm2_debug_enabled()) {
        PySys_WriteStderr("[sm2_ext] sign: end\n");
    }
    return sig;
}

static PyMethodDef Sm2Methods[] = {
    {"sign", sm2_sign, METH_VARARGS, "Sign message (SM2) - C backend stub."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sm2module = {
    PyModuleDef_HEAD_INIT,
    "sm2_ext",
    "CPython SM2 backend (stub)",
    -1,
    Sm2Methods
};

PyMODINIT_FUNC PyInit_sm2_ext(void) {
    if (PyThread_tss_create(&sm2_cache_key) != 0) {
        return NULL;
    }
    return PyModule_Create(&sm2module);
}
