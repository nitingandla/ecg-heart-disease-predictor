import numpy as np

def split_into_leads(img):
    h, w = img.shape

    rows, cols = 3, 4
    h_step = h // rows
    w_step = w // cols

    leads = []

    for i in range(rows):
        for j in range(cols):
            lead = img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]

            lead = lead / 255.0

            leads.append(lead)

    return leads