/**
 * layers.c - Layer composition for domain adaptation ("HAL memory cartridge")
 *
 * Enables hot-swapping of domain-specific overlays on top of a base model.
 */

#define _POSIX_C_SOURCE 200809L

#include "../psam_internal.h"
#include <stdlib.h>
#include <string.h>

/* ============================ Layer Management ============================ */

psam_error_t psam_add_layer(
    psam_model_t* base,
    const char* layer_id,
    psam_model_t* overlay,
    float weight
) {
    if (!base || !layer_id || !overlay) {
        return PSAM_ERR_NULL_PARAM;
    }

    /* Validate compatibility */
    if (base->config.vocab_size != overlay->config.vocab_size) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    if (!overlay->is_finalized) {
        return PSAM_ERR_NOT_TRAINED;
    }

    /* Check for duplicate layer ID */
    psam_lock_rdlock(&base->lock);
    layer_node_t* node = base->layers;
    while (node) {
        if (strncmp(node->id, layer_id, 64) == 0) {
            psam_lock_unlock_rd(&base->lock);
            return PSAM_ERR_INVALID_CONFIG;  /* Duplicate ID */
        }
        node = node->next;
    }
    psam_lock_unlock_rd(&base->lock);

    /* Create new layer node */
    layer_node_t* new_layer = calloc(1, sizeof(layer_node_t));
    if (!new_layer) {
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    strncpy(new_layer->id, layer_id, 63);
    new_layer->id[63] = '\0';
    new_layer->model = overlay;
    new_layer->weight = weight;

    /* Add to front of list (exclusive lock) */
    psam_lock_wrlock(&base->lock);
    new_layer->next = base->layers;
    base->layers = new_layer;
    psam_lock_unlock_wr(&base->lock);

    return PSAM_OK;
}

psam_error_t psam_remove_layer(psam_model_t* base, const char* layer_id) {
    if (!base || !layer_id) {
        return PSAM_ERR_NULL_PARAM;
    }

    psam_lock_wrlock(&base->lock);

    layer_node_t* prev = NULL;
    layer_node_t* curr = base->layers;

    while (curr) {
        if (strncmp(curr->id, layer_id, 64) == 0) {
            /* Found it - remove from list */
            if (prev) {
                prev->next = curr->next;
            } else {
                base->layers = curr->next;
            }

            psam_lock_unlock_wr(&base->lock);
            free(curr);
            return PSAM_OK;
        }

        prev = curr;
        curr = curr->next;
    }

    psam_lock_unlock_wr(&base->lock);

    return PSAM_ERR_LAYER_NOT_FOUND;
}

psam_error_t psam_update_layer_weight(psam_model_t* base, const char* layer_id, float new_weight) {
    if (!base || !layer_id) {
        return PSAM_ERR_NULL_PARAM;
    }

    psam_lock_wrlock(&base->lock);

    layer_node_t* node = base->layers;
    while (node) {
        if (strncmp(node->id, layer_id, 64) == 0) {
            node->weight = new_weight;
            psam_lock_unlock_wr(&base->lock);
            return PSAM_OK;
        }
        node = node->next;
    }

    psam_lock_unlock_wr(&base->lock);

    return PSAM_ERR_LAYER_NOT_FOUND;
}

int psam_list_layers(psam_model_t* model, const char** out_ids, size_t max_layers) {
    if (!model || !out_ids) {
        return PSAM_ERR_NULL_PARAM;
    }

    psam_lock_rdlock(&model->lock);

    int count = 0;
    layer_node_t* node = model->layers;

    while (node && count < (int)max_layers) {
        out_ids[count] = node->id;
        count++;
        node = node->next;
    }

    psam_lock_unlock_rd(&model->lock);

    return count;
}
