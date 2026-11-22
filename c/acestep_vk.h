/* acestep_vk.h: the Vulkan device the ACE-Step engine's parts share (the planner, acestep_lm_vk.h, and the DiT, acestep_dit_vk.h).
 *
 * Opened the first time a part asks: COLI_VULKAN unset or no device leaves every part on the CPU; COLI_VK_CHAIN=0 does too. A device that is lost
 * takes every part off the card for the rest of the process (acv_lost). */
#ifndef ACESTEP_VK_H
#define ACESTEP_VK_H
#ifdef COLI_VULKAN

#include <stdlib.h>
#include "backend_vulkan.h"
#include "vk_chain.h"

static struct { int tried, chain; } acv_dev;

/* 1 when the dense chain is up. */
static int acv_open(void){
    if (!acv_dev.tried) {
        acv_dev.tried = 1;
        if (coli_vk_init_env("acestep")) {                    /* COLI_VULKAN unset, or no device: the CPU */
            atexit(coli_vk_shutdown);
            const char *c = getenv("COLI_VK_CHAIN");
            if (!(c && !strcmp(c, "0")) && vkc_init() && vkc_ready()) { acv_dev.chain = 1; atexit(vkc_shutdown); }   /* registered after the backend's: runs before it */
        }
    }
    return acv_dev.chain;
}

/* After a failed submission: a lost device is lost for everyone. */
static void acv_failed(void){
    if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); acv_dev.chain = 0; }
}

#endif
#endif
