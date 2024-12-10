import numpy as np
import random
import cv2
import torchvision.transforms as transforms
from ultralytics import YOLO
import torch
from torchvision import models
import torch.nn as nn
import os
import logging
from Pipeline.Mapper import map_number_to_ship
from Pipeline.ship_image_scraper import scrape_image_selenium as scrape_image  # Selenium-basierte Funktion importieren
import time

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_color():
    return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

def save_image(image_np, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)
    return image_path

def process_image(image, yolo_model, detection_labels, model_classification, device, process_id):
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Führe YOLO-Detektion und Segmentierung durch
    yolo_result = yolo_model.predict(image_np, task='segment')
    detected_boxes = 0
    boxes_data = []
    drawn_masks = []

    detections = yolo_result[0]

    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        confidence = box.conf[0]

        if confidence > 0.64 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image.crop((x1, y1, x2, y2))
            label_pred = classify_and_log(cropped_image, model_classification, device)

            label_text = f'{label_pred} {confidence:.2f}'
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            boxes_data.append({
                'label': label_pred,
                'confidence': confidence,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            })

            if detections.masks is not None:
                mask = detections.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized.astype(bool)

                color = generate_unique_color()
                image_np[mask_resized] = (image_np[mask_resized] * 0.5 + color * 0.5).astype(np.uint8)

                drawn_masks.append((confidence, label_pred, mask_resized))

    if detected_boxes == 0:
        logger.warning("Keine Objekte in YOLO-Detektion gefunden.")
        return None, None

    # Speichere das verarbeitete Bild und die XML-Daten
    image_path = save_image(image_np, "output2", f"{process_id}_processed.jpg")

    return image_path

def classify_image(image, model_classification, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_classification(image_tensor)
        _, preds = torch.max(outputs, 1)

    return preds.item()

def classify_and_log(image, model_classification, device):
    label_pred = map_number_to_ship(classify_image(image, model_classification, device))
    logger.info(f"Schiffkategorie ist : {label_pred}")
    return label_pred

def process_single_image(process_id, yolo_model, detection_labels, model_classification, device):
    image = scrape_image(process_id)
    if image is None:
        return None, None

    image_path = process_image(image, yolo_model, detection_labels, model_classification, device, process_id)
    return image_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    yolo_model = YOLO('/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/optuna_trial_1/weights/best.pt')
    # Überprüfen Sie, ob YOLO das Gerät setzen kann
    if hasattr(yolo_model, 'to'):
        yolo_model.to(device)
    else:
        logger.warning("YOLO-Modell unterstützt die `.to(device)` Methode nicht.")

    # Klassifikationsmodell laden
    num_classes = 9
    model_classification = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_classification.fc.in_features
    model_classification.fc = nn.Linear(num_ftrs, num_classes)

    # Pfad zu den gespeicherten Gewichten
    model_save_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/best_model_trial_2.pth"
    if not os.path.exists(model_save_path):
        logger.error(f"Klassifikationsmodell nicht gefunden am Pfad: {model_save_path}")
        return

    state_dict = torch.load(model_save_path, map_location=device)
    model_classification.load_state_dict(state_dict)
    model_classification.to(device)
    model_classification.eval()

    detection_labels = ["boat"]
    process_ids = [
        15001, 15002, 15003, 15004, 15005, 15006, 15007, 15008, 15009, 15010,
        15011, 15012, 15013, 15014, 15015, 15016, 15017, 15018, 15019, 15020,
        15021, 15022, 15023, 15024, 15025, 15026, 15027, 15028, 15029, 15030,
        15031, 15032, 15033, 15034, 15035, 15036, 15037, 15038, 15039, 15040,
        15041, 15042, 15043, 15044, 15045, 15046, 15047, 15048, 15049, 15050,
        15051, 15052, 15053, 15054, 15055, 15056, 15057, 15058, 15059, 15060,
        15061, 15062, 15063, 15064, 15065, 15066, 15067, 15068, 15069, 15070,
        15071, 15072, 15073, 15074, 15075, 15076, 15077, 15078, 15079, 15080,
        15081, 15082, 15083, 15084, 15085, 15086, 15087, 15088, 15089, 15090,
        15091, 15092, 15093, 15094, 15095, 15096, 15097, 15098, 15099, 15100,
        15101, 15102, 15103, 15104, 15105, 15106, 15107, 15108, 15109, 15110,
        15111, 15112, 15113, 15114, 15115, 15116, 15117, 15118, 15119, 15120,
        15121, 15122, 15123, 15124, 15125, 15126, 15127, 15128, 15129, 15130,
        15131, 15132, 15133, 15134, 15135, 15136, 15137, 15138, 15139, 15140,
        15141, 15142, 15143, 15144, 15145, 15146, 15147, 15148, 15149, 15150,
        15151, 15152, 15153, 15154, 15155, 15156, 15157, 15158, 15159, 15160,
        15161, 15162, 15163, 15164, 15165, 15166, 15167, 15168, 15169, 15170,
        15171, 15172, 15173, 15174, 15175, 15176, 15177, 15178, 15179, 15180,
        15181, 15182, 15183, 15184, 15185, 15186, 15187, 15188, 15189, 15190,
        15191, 15192, 15193, 15194, 15195, 15196, 15197, 15198, 15199, 15200,
        15201, 15202, 15203, 15204, 15205, 15206, 15207, 15208, 15209, 15210,
        15211, 15212, 15213, 15214, 15215, 15216, 15217, 15218, 15219, 15220,
        15221, 15222, 15223, 15224, 15225, 15226, 15227, 15228, 15229, 15230,
        15231, 15232, 15233, 15234, 15235, 15236, 15237, 15238, 15239, 15240,
        15241, 15242, 15243, 15244, 15245, 15246, 15247, 15248, 15249, 15250,
        15251, 15252, 15253, 15254, 15255, 15256, 15257, 15258, 15259, 15260,
        15261, 15262, 15263, 15264, 15265, 15266, 15267, 15268, 15269, 15270,
        15271, 15272, 15273, 15274, 15275, 15276, 15277, 15278, 15279, 15280,
        15281, 15282, 15283, 15284, 15285, 15286, 15287, 15288, 15289, 15290,
        15291, 15292, 15293, 15294, 15295, 15296, 15297, 15298, 15299, 15300,
        15301, 15302, 15303, 15304, 15305, 15306, 15307, 15308, 15309, 15310,
        15311, 15312, 15313, 15314, 15315, 15316, 15317, 15318, 15319, 15320,
        15321, 15322, 15323, 15324, 15325, 15326, 15327, 15328, 15329, 15330,
        15331, 15332, 15333, 15334, 15335, 15336, 15337, 15338, 15339, 15340,
        15341, 15342, 15343, 15344, 15345, 15346, 15347, 15348, 15349, 15350,
        15351, 15352, 15353, 15354, 15355, 15356, 15357, 15358, 15359, 15360,
        15361, 15362, 15363, 15364, 15365, 15366, 15367, 15368, 15369, 15370,
        15371, 15372, 15373, 15374, 15375, 15376, 15377, 15378, 15379, 15380,
        15381, 15382, 15383, 15384, 15385, 15386, 15387, 15388, 15389, 15390,
        15391, 15392, 15393, 15394, 15395, 15396, 15397, 15398, 15399, 15400,
        15401, 15402, 15403, 15404, 15405, 15406, 15407, 15408, 15409, 15410,
        15411, 15412, 15413, 15414, 15415, 15416, 15417, 15418, 15419, 15420,
        15421, 15422, 15423, 15424, 15425, 15426, 15427, 15428, 15429, 15430,
        15431, 15432, 15433, 15434, 15435, 15436, 15437, 15438, 15439, 15440,
        15441, 15442, 15443, 15444, 15445, 15446, 15447, 15448, 15449, 15450,
        15451, 15452, 15453, 15454, 15455, 15456, 15457, 15458, 15459, 15460,
        15461, 15462, 15463, 15464, 15465, 15466, 15467, 15468, 15469, 15470,
        15471, 15472, 15473, 15474, 15475, 15476, 15477, 15478, 15479, 15480,
        15481, 15482, 15483, 15484, 15485, 15486, 15487, 15488, 15489, 15490,
        15491, 15492, 15493, 15494, 15495, 15496, 15497, 15498, 15499, 15500,
        15501, 15502, 15503, 15504, 15505, 15506, 15507, 15508, 15509, 15510,
        15511, 15512, 15513, 15514, 15515, 15516, 15517, 15518, 15519, 15520,
        15521, 15522, 15523, 15524, 15525, 15526, 15527, 15528, 15529, 15530,
        15531, 15532, 15533, 15534, 15535, 15536, 15537, 15538, 15539, 15540,
        15541, 15542, 15543, 15544, 15545, 15546, 15547, 15548, 15549, 15550,
        15551, 15552, 15553, 15554, 15555, 15556, 15557, 15558, 15559, 15560,
        15561, 15562, 15563, 15564, 15565, 15566, 15567, 15568, 15569, 15570,
        15571, 15572, 15573, 15574, 15575, 15576, 15577, 15578, 15579, 15580,
        15581, 15582, 15583, 15584, 15585, 15586, 15587, 15588, 15589, 15590,
        15591, 15592, 15593, 15594, 15595, 15596, 15597, 15598, 15599, 15600,
        15601, 15602, 15603, 15604, 15605, 15606, 15607, 15608, 15609, 15610,
        15611, 15612, 15613, 15614, 15615, 15616, 15617, 15618, 15619, 15620,
        15621, 15622, 15623, 15624, 15625, 15626, 15627, 15628, 15629, 15630,
        15631, 15632, 15633, 15634, 15635, 15636, 15637, 15638, 15639, 15640,
        15641, 15642, 15643, 15644, 15645, 15646, 15647, 15648, 15649, 15650,
        15651, 15652, 15653, 15654, 15655, 15656, 15657, 15658, 15659, 15660,
        15661, 15662, 15663, 15664, 15665, 15666, 15667, 15668, 15669, 15670,
        15671, 15672, 15673, 15674, 15675, 15676, 15677, 15678, 15679, 15680,
        15681, 15682, 15683, 15684, 15685, 15686, 15687, 15688, 15689, 15690,
        15691, 15692, 15693, 15694, 15695, 15696, 15697, 15698, 15699, 15700,
        15701, 15702, 15703, 15704, 15705, 15706, 15707, 15708, 15709, 15710,
        15711, 15712, 15713, 15714, 15715, 15716, 15717, 15718, 15719, 15720,
        15721, 15722, 15723, 15724, 15725, 15726, 15727, 15728, 15729, 15730,
        15731, 15732, 15733, 15734, 15735, 15736, 15737, 15738, 15739, 15740,
        15741, 15742, 15743, 15744, 15745, 15746, 15747, 15748, 15749, 15750,
        15751, 15752, 15753, 15754, 15755, 15756, 15757, 15758, 15759, 15760,
        15761, 15762, 15763, 15764, 15765, 15766, 15767, 15768, 15769, 15770,
        15771, 15772, 15773, 15774, 15775, 15776, 15777, 15778, 15779, 15780,
        15781, 15782, 15783, 15784, 15785, 15786, 15787, 15788, 15789, 15790,
        15791, 15792, 15793, 15794, 15795, 15796, 15797, 15798, 15799, 15800,
        15801, 15802, 15803, 15804, 15805, 15806, 15807, 15808, 15809, 15810,
        15811, 15812, 15813, 15814, 15815, 15816, 15817, 15818, 15819, 15820,
        15821, 15822, 15823, 15824, 15825, 15826, 15827, 15828, 15829, 15830,
        15831, 15832, 15833, 15834, 15835, 15836, 15837, 15838, 15839, 15840,
        15841, 15842, 15843, 15844, 15845, 15846, 15847, 15848, 15849, 15850,
        15851, 15852, 15853, 15854, 15855, 15856, 15857, 15858, 15859, 15860,
        15861, 15862, 15863, 15864, 15865, 15866, 15867, 15868, 15869, 15870,
        15871, 15872, 15873, 15874, 15875, 15876, 15877, 15878, 15879, 15880,
        15881, 15882, 15883, 15884, 15885, 15886, 15887, 15888, 15889, 15890,
        15891, 15892, 15893, 15894, 15895, 15896, 15897, 15898, 15899, 15900,
        15901, 15902, 15903, 15904, 15905, 15906, 15907, 15908, 15909, 15910,
        15911, 15912, 15913, 15914, 15915, 15916, 15917, 15918, 15919, 15920,
        15921, 15922, 15923, 15924, 15925, 15926, 15927, 15928, 15929, 15930,
        15931, 15932, 15933, 15934, 15935, 15936, 15937, 15938, 15939, 15940,
        15941, 15942, 15943, 15944, 15945, 15946, 15947, 15948, 15949, 15950,
        15951, 15952, 15953, 15954, 15955, 15956, 15957, 15958, 15959, 15960,
        15961, 15962, 15963, 15964, 15965, 15966, 15967, 15968, 15969, 15970,
        15971, 15972, 15973, 15974, 15975, 15976, 15977, 15978, 15979, 15980,
        15981, 15982, 15983, 15984, 15985, 15986, 15987, 15988, 15989, 15990,
        15991, 15992, 15993, 15994, 15995, 15996, 15997, 15998, 15999, 16000,
    ]

    for process_id in process_ids:
        logger.info(f"Verarbeite process_id: {process_id}")
        process_single_image(process_id, yolo_model, detection_labels, model_classification, device)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
