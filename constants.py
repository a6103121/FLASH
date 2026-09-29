IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_LENGTH = 576
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

INSTRUCTION_TEMPLATE = {
    "llava-1.5": "USER: <ImageHere> <question>"
}

INSTRUCTION_TEMPLATE_NO_IMG = {
    "llava-1.5": "USER: <question>"
}

SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
POPE_CHAT_PATH = {
    "random": "./pope_coco/chat/coco_pope_chat_random.json",
    "popular": "./pope_coco/chat/coco_pope_chat_popular.json",
    "adversarial": "./pope_coco/chat/coco_pope_chat_adversarial.json"
}
