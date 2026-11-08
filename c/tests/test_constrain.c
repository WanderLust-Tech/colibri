/* constrain.h, on a fake vocabulary of ASCII bytes and a few multi-byte tokens, with random logits at every step.
 *
 *   test_constrain                       the checks; exit 0 when they all hold
 *   test_constrain gen generic N SEED    N random constrained generations under the generic JSON grammar
 *   test_constrain gen schema N SEED     the same under a JSON schema
 *
 * `gen` prints one line per generation: "C <hex>" when the answer ended by itself, "T <hex>" when it hit the token limit.
 * tests/test_constrain.py reads them with a real JSON parser, the oracle that is not this tree's own grammar engine. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../constrain.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static uint64_t rng_s = 88172645463325252ULL;
static double rnd(void){ rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (rng_s >> 11) * (1.0 / 9007199254740992.0); }

enum { N_ASCII = 95, T_FIRST_MULTI = 95, T_EOS = 112, T_SPECIAL = 113, V = 114 };
static const char *MULTI[] = {"{\"", "\":", "true", "false", "null", "{}", "[]", "\",", "\"}", "\\n", "12", "\n", "\t",
                              "  ", "\\\"", "\":1", "1}"};   /* 17 tokens, ids 95..111 */

static CnVocab *make_vocab(int with_digits){
    CnVocab *v = cn_vocab_new(V);
    for (int t = 0; t < N_ASCII; t++){
        unsigned char b = (unsigned char)(0x20 + t);
        if (!with_digits && b >= '0' && b <= '9'){ cn_vocab_set(v, t, NULL, 0, CN_SPECIAL); continue; }
        cn_vocab_set(v, t, &b, 1, CN_TEXT);
    }
    for (int i = 0; i < 17; i++){
        if (!with_digits && strpbrk(MULTI[i], "0123456789")){ cn_vocab_set(v, T_FIRST_MULTI + i, NULL, 0, CN_SPECIAL); continue; }
        cn_vocab_set(v, T_FIRST_MULTI + i, (const unsigned char *)MULTI[i], (int)strlen(MULTI[i]), CN_TEXT);
    }
    cn_vocab_set(v, T_EOS, NULL, 0, CN_END);
    cn_vocab_set(v, T_SPECIAL, (const unsigned char *)"<s>", 3, CN_SPECIAL);
    return v;
}

static int argmax(const float *lo, void *ctx){
    (void)ctx;
    int b = 0;
    for (int i = 1; i < V; i++) if (lo[i] > lo[b]) b = i;
    return b;
}
static int always_close_brace(const float *lo, void *ctx){ (void)lo; (void)ctx; return '}' - 0x20; }
static int first_multi(const float *lo, void *ctx){ (void)lo; (void)ctx; return T_FIRST_MULTI + 1; }   /* ":  always */
static int counting_argmax(const float *lo, void *ctx){ ++*(int *)ctx; return argmax(lo, NULL); }

static const char *GENERIC =
    "root ::= jws jval jws\n"
    "jval ::= jobj | jarr | jstr | jnum | \"true\" | \"false\" | \"null\"\n"
    "jobj ::= \"{\" jws ( jstr jws \":\" jws jval jws ( \",\" jws jstr jws \":\" jws jval jws )* )? \"}\"\n"
    "jarr ::= \"[\" jws ( jval jws ( \",\" jws jval jws )* )? \"]\"\n"
    "jstr ::= \"\\\"\" jchar* \"\\\"\"\n"
    "jchar ::= [^\"\\\\\\x00-\\x1f] | \"\\\\\" ( [\"\\\\/bfnrt] | \"u\" jhex jhex jhex jhex )\n"
    "jhex ::= [0-9a-fA-F]\n"
    "jnum ::= \"-\"? ( \"0\" | [1-9] [0-9]* ) ( \".\" [0-9]+ )? ( ( \"e\" | \"E\" ) ( \"+\" | \"-\" )? [0-9]+ )?\n"
    "jws ::= ( \" \" | \"\\t\" | \"\\n\" | \"\\r\" )*\n";

static const char *SCHEMA =
    "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"},"
    "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"ok\":{\"type\":\"boolean\"}},"
    "\"required\":[\"name\",\"n\",\"tags\",\"ok\"]}";

/* one generation with random logits, a third of the steps tempting the model to stop. Returns the token count; the bytes go to
 * out; *ended says whether the answer closed itself. -1: a request the vocabulary cannot continue. */
static int generate(Cn *c, const CnVocab *v, unsigned char *out, int cap, int max_tok, int *ended, CnSampler pick){
    int n = 0; float lo[V];
    *ended = 0;
    for (int step = 0; step < max_tok; step++){
        for (int t = 0; t < V; t++) lo[t] = (float)rnd();
        if (rnd() < 0.33) lo[T_EOS] = 2.0f;
        int t = cn_pick(c, v, lo, pick, NULL);
        if (t < 0) return -1;
        if (v->kind[t] == CN_END){ *ended = 1; break; }
        for (int i = 0; i < v->len[t] && n < cap; i++) out[n++] = v->pool[v->off[t] + i];
    }
    out[n] = 0;
    return n;
}

/* the output, fed to a fresh walker, is accepted whole; when it ended, the parse could end there */
static int replays(const char *gbnf_or_schema, const unsigned char *s, int n, int ended){
    char err[160];
    Cn *c = cn_new(gbnf_or_schema, err, sizeof err);
    if (!c) return 0;
    int ok = 1;
    for (int i = 0; i < n && ok; i++) ok = gr_accept(&c->st, s[i]) == 1;
    if (ok && ended){ unsigned char m[32]; int end; gr_admissible(&c->st, m, &end); ok = end; }
    cn_free(c);
    return ok;
}

static int run_gen(const char *which, int count, uint64_t seed){
    const char *payload = !strcmp(which, "schema") ? SCHEMA : GENERIC;
    CnVocab *v = make_vocab(1);
    rng_s ^= seed * 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < count; i++){
        char err[160];
        Cn *c = cn_new(payload, err, sizeof err);
        if (!c){ fprintf(stderr, "cn_new: %s\n", err); return 1; }
        unsigned char out[2048]; int ended;
        int n = generate(c, v, out, 2000, 90, &ended, argmax);
        if (n < 0){ fprintf(stderr, "generation %d: no token could continue\n", i); return 1; }
        printf("%c ", ended ? 'C' : 'T');
        for (int k = 0; k < n; k++) printf("%02x", out[k]);
        printf("\n");
        cn_free(c);
    }
    cn_vocab_free(v);
    return 0;
}

int main(int argc, char **argv){
    if (argc >= 5 && !strcmp(argv[1], "gen")) return run_gen(argv[2], atoi(argv[3]), (uint64_t)atoll(argv[4]));
    char err[160];
    CnVocab *v = make_vocab(1);

    /* every random generation stays inside the generic JSON grammar, ended or cut off */
    int ended_n = 0;
    for (int i = 0; i < 400; i++){
        Cn *c = cn_new(GENERIC, err, sizeof err);
        CHECK(c);
        unsigned char out[2048]; int ended;
        int n = generate(c, v, out, 2000, 80, &ended, argmax);
        CHECK(n >= 0);
        CHECK(replays(GENERIC, out, n, ended));
        ended_n += ended;
        cn_free(c);
    }
    CHECK(ended_n > 100);                                   /* the test must also reach complete answers */

    /* and inside a schema */
    for (int i = 0; i < 200; i++){
        Cn *c = cn_new(SCHEMA, err, sizeof err);
        CHECK(c);
        unsigned char out[2048]; int ended;
        int n = generate(c, v, out, 2000, 120, &ended, argmax);
        CHECK(n >= 0);
        CHECK(replays(SCHEMA, out, n, ended));
        if (ended){                                         /* a whole object: first and last byte that is not space are the braces */
            int a = 0, b = n - 1;
            while (a < n && strchr(" \t\n\r", out[a])) a++;
            while (b > a && strchr(" \t\n\r", out[b])) b--;
            CHECK(a < b && out[a] == '{' && out[b] == '}');
        }
        cn_free(c);
    }

    /* a grammar that has ended leaves only the end: every other token is masked, and the end is chosen */
    {
        Cn *c = cn_new("root ::= \"ok\"", err, sizeof err);
        CHECK(c);
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 1.0f;
        CHECK(cn_pick(c, v, lo, argmax, NULL) == 'o' - 0x20);
        for (int t = 0; t < V; t++) lo[t] = 1.0f;
        CHECK(cn_pick(c, v, lo, argmax, NULL) == 'k' - 0x20);
        for (int t = 0; t < V; t++) lo[t] = 1.0f;
        lo[0] = 5.0f;                                       /* a space is far more likely, and not allowed */
        CHECK(cn_pick(c, v, lo, argmax, NULL) == T_EOS);
        cn_free(c);
    }

    /* the end is not allowed before the grammar can end */
    {
        Cn *c = cn_new("root ::= \"ab\"", err, sizeof err);
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 0.0f;
        lo[T_EOS] = 9.0f;
        CHECK(cn_pick(c, v, lo, argmax, NULL) == 'a' - 0x20);
        cn_free(c);
    }

    /* a token that is valid in its first byte and wrong in its second is refused: after `{`, the token `"}` is not `"` then a key */
    {
        Cn *c = cn_new("root ::= \"{\\\"\" [a-z]+ \"\\\":1}\"", err, sizeof err);
        CHECK(c);
        CHECK(cn_valid(c, v, T_FIRST_MULTI + 0));            /* {" */
        CHECK(!cn_valid(c, v, T_FIRST_MULTI + 5));           /* {} : the `}` has no place */
        CHECK(!cn_valid(c, v, T_EOS));
        CHECK(!cn_valid(c, v, T_SPECIAL));
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 0.0f;
        lo[T_FIRST_MULTI + 5] = 9.0f;                        /* the sampler wants {} */
        lo[T_FIRST_MULTI + 0] = 1.0f;
        CHECK(cn_pick(c, v, lo, argmax, NULL) == T_FIRST_MULTI + 0);
        cn_free(c);
    }

    /* a sampler that insists on an invalid token is overruled by a scan for the best valid one */
    {
        Cn *c = cn_new("root ::= \"ab\"", err, sizeof err);
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 0.0f;
        lo['a' - 0x20] = 0.5f;
        CHECK(cn_pick(c, v, lo, always_close_brace, NULL) == 'a' - 0x20);
        cn_free(c);
    }

    /* the scan must itself check: three tokens that begin right and go wrong outrank the one valid token, and the sampler only
     * ever names the first of them */
    {
        Cn *c = cn_new("root ::= \"\\\"x\"", err, sizeof err);
        CHECK(c);
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 0.0f;
        lo[T_FIRST_MULTI + 1] = 9.0f;                        /* ":   */
        lo[T_FIRST_MULTI + 7] = 8.0f;                        /* ",   */
        lo[T_FIRST_MULTI + 8] = 7.0f;                        /* "}   */
        lo['"' - 0x20] = 1.0f;
        CHECK(cn_pick(c, v, lo, first_multi, NULL) == '"' - 0x20);
        cn_free(c);
    }

    /* a token that failed is not offered again: with five that fail, the sampler is asked six times, not sixty-four */
    {
        Cn *c = cn_new("root ::= \"\\\"x\"", err, sizeof err);
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 0.0f;
        int calls = 0;
        lo[T_FIRST_MULTI + 1] = 9.0f; lo[T_FIRST_MULTI + 7] = 8.0f; lo[T_FIRST_MULTI + 8] = 7.0f;
        lo[T_FIRST_MULTI + 15] = 6.0f;                       /* ":1  */
        lo['"' - 0x20] = 1.0f;
        CHECK(cn_pick(c, v, lo, counting_argmax, &calls) == '"' - 0x20);
        CHECK(calls == 5);                                   /* four that fail, then the valid one */
        cn_free(c);
    }

    /* a walker that was lost is an error, not an unconstrained pick */
    {
        Cn *c = cn_new("root ::= \"ab\"", err, sizeof err);
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 1.0f;
        c->st.alive = 0;
        CHECK(cn_pick(c, v, lo, argmax, NULL) == -2);
        CHECK(!cn_valid(c, v, 'a' - 0x20));
        cn_free(c);
    }

    /* a grammar that needs a byte no token carries cannot be continued: -1, never an unconstrained token */
    {
        CnVocab *nodigits = make_vocab(0);
        Cn *c = cn_new("root ::= [0-9]+", err, sizeof err);
        float lo[V];
        for (int t = 0; t < V; t++) lo[t] = 1.0f;
        CHECK(cn_pick(c, nodigits, lo, argmax, NULL) == -1);
        cn_free(c);
        cn_vocab_free(nodigits);
    }

    /* what does not compile is refused, with a reason */
    err[0] = 0;
    CHECK(cn_new("root ::= ( \"a\" ", err, sizeof err) == NULL && err[0]);      /* an open group */
    CHECK(cn_new("nothing here", err, sizeof err) == NULL);
    CHECK(cn_new("other ::= \"a\"", err, sizeof err) == NULL);                   /* no root rule */
    CHECK(cn_new("{\"anyOf\":[{\"type\":\"string\"}]}", err, sizeof err) == NULL && err[0]);   /* outside the schema subset */
    CHECK(cn_new("{not json", err, sizeof err) == NULL);

    cn_vocab_free(v);
    printf("constrain: ok (%d of 400 generic generations ended by themselves)\n", ended_n);
    return 0;
}
