"""Fixed hypotheses for a development-only semantic-prior control.

These strings are not diagnoses or new image labels. Real clinical labels and
unchanged app-verdict gates determine whether the control has any value.
"""
PROMPTS=[
    ('healthy_upper','a photo of a healthy tomato leaf, uniform green tissue, no lesions, no mines, no mold, and no damaged edges.'),
    ('healthy_lower','a photo of the pale green underside of a healthy tomato leaf, visible veins, without spots or damage.'),
    ('healthy_young','a photo of a healthy young light green tomato leaf with no visible disease or damage.'),
    ('early_blight','a photo of a tomato leaf with early blight, brown circular lesions with rings and yellow halos.'),
    ('late_blight','a photo of a tomato leaf with late blight, irregular dark brown damaged patches.'),
    ('leaf_miner','a photo of a tomato leaf with leaf miner damage, winding pale trails through the leaf.'),
    ('magnesium','a photo of a tomato leaf with magnesium deficiency, pale yellow tissue between green veins.'),
    ('nitrogen','a photo of a tomato leaf with nitrogen deficiency, widespread yellowing of an older leaf.'),
    ('potassium','a photo of a tomato leaf with potassium deficiency, yellow and brown damaged leaf edges.'),
    ('spotted_wilt','a photo of a tomato leaf with spotted wilt, bronze tissue, dark spots and ring patterns.'),
    ('bacterial_spot','a photo of a tomato leaf with bacterial spot, small dark lesions and yellow surrounding tissue.'),
    ('septoria','a photo of a tomato leaf with many small round brown spots with pale centers.'),
    ('mites','a photo of a tomato leaf with spider mite damage, fine pale stippling, bronzing and thin webbing.'),
    ('leaf_mold','a photo of a diseased tomato leaf with leaf mold, yellow patches and mold underneath.'),
    ('powdery_mildew','a photo of a tomato leaf with white powdery mildew on its surface.'),
    ('leaf_curl','a photo of a diseased tomato leaf with abnormal curling, distorted growth and yellowing.'),
    ('target_spot','a photo of a tomato leaf with numerous brown lesions and concentric target-like rings.'),
    ('damaged','a photo of a visibly damaged tomato leaf with dead tissue, lesions or infestation.'),
    ('not_tomato','a photo of a leaf from a plant other than tomato.'),
]
