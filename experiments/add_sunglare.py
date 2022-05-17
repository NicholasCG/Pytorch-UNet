import cv2
import numpy as np
import pandas as pd
import asyncio

path = '/media/lumi/Backups/OneDrive/LISA_Dataset/signDatabasePublicFramesOnly/'

all = pd.read_csv(path + 'allAnnotations.csv', sep=';')

images = all['Filename']
# images = ["vid7/frameAnnotations-MVI_0119.MOV_annotations/speedLimit_1324865659.avi_image7.png"]

glare1 = cv2.imread('glare.png')
glare1 = cv2.cvtColor(glare1, cv2.COLOR_BGR2GRAY)
glare1 = cv2.cvtColor(glare1, cv2.COLOR_GRAY2RGB)

glare2 = cv2.imread('glare2.png', cv2.IMREAD_UNCHANGED)
glare3 = cv2.imread('glare3.png', cv2.IMREAD_UNCHANGED)

glare2_alpha = glare2[:, :, 3]
glare2_alpha = np.reshape(glare2_alpha, (glare2_alpha.shape[0], glare2_alpha.shape[1], 1))

glare2 = cv2.cvtColor(glare2, cv2.COLOR_BGR2GRAY)
glare2 = cv2.cvtColor(glare2, cv2.COLOR_GRAY2RGB)
glare2 = np.concatenate((glare2, glare2_alpha), axis=2)

glare3_alpha = glare3[:, :, 3]
glare3_alpha = np.reshape(glare3_alpha, (glare3_alpha.shape[0], glare3_alpha.shape[1], 1))

glare3 = cv2.cvtColor(glare3, cv2.COLOR_BGR2GRAY)
glare3 = cv2.cvtColor(glare3, cv2.COLOR_GRAY2RGB)
glare3 = np.concatenate((glare3, glare3_alpha), axis=2)

# cv2.imshow('glare1', glare1)
# cv2.imshow('glare2', glare2)
# cv2.imshow('glare3', glare3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

async def main():
    await asyncio.gather(*(add_glare(img_num+1, image) for img_num, image in enumerate(images)))
    # await add_glare(0, images)

async def add_glare(img_num, image):
    print(f'{img_num}/{len(images)}')
    img = cv2.imread(path + image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    zeros_img = np.full((img.shape[0], img.shape[1], 1), fill_value = 255, dtype=np.uint8)
    img = np.concatenate((img, zeros_img), axis=2)

    # Adding glare1 to the image

    for x in range(3):
        img1 = img.copy()[:, :, :3]
        glare1_copy = cv2.resize(glare1, (img.shape[1]*2, img.shape[0]*2))
        tx = np.random.randint(img.shape[1] * 0.2, img.shape[1] * 0.8)
        ty = np.random.randint(img.shape[0]/2, img.shape[0] * 0.7)

        glare1_copy = glare1_copy[ty:ty+img.shape[0], tx:tx+img.shape[1]]

        gamma = 0.7
        # gamma = 1
        img1 = cv2.addWeighted(img1, 1, glare1_copy, gamma, 0)

        # test = img1 - img[:,:,:3]
        # cv2.imshow('test', test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # print(f"residual {test[58,344, :]}")
        # print(f"glare {img1[58, 344, :]}")
        cv2.imwrite(f'../data/imgs/img{img_num}_glare{x+1}.png', img1[:,:,:3])
        # cv2.imwrite(f'./red/img{img_num}_glare{x+1}_mask.png', (img1 - img[:,:,:3])[:, :, 2])
        # cv2.imwrite(f'./green/img{img_num}_glare{x+1}_mask.png', (img1 - img[:,:,:3])[:, :, 1])
        # cv2.imwrite(f'./blue/img{img_num}_glare{x+1}_mask.png', (img1 - img[:,:,:3])[:, :, 0])
        cv2.imwrite(f'../data/masks/img{img_num}_glare{x+1}_mask.png', (img1 - img[:,:,:3]))

    # Adding glare2 to the image
    for x in range(3, 6):
        img2 = img.copy()
        glare2_copy = cv2.resize(glare2, (img.shape[1], img.shape[0]))

        height, width = img.shape[:2]
        tx = np.random.randint(img.shape[1] * -0.4, img.shape[1] * 0.4)
        ty = np.random.randint(img.shape[0] * -0.2, img.shape[0] * 0.1)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        glare2_copy = cv2.warpAffine(glare2_copy, translation_matrix, (width, height))

        alpha_background = img2[:, :, 3] / 255.0
        alpha_foreground = glare2_copy[:, :, 3] / 255.0

        for color in range(3):
            img2[:,:,color] = alpha_foreground * glare2_copy[:,:,color] + \
            alpha_background * img2[:,:,color] * (1 - alpha_foreground)

        img2[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
        cv2.imwrite(f'../data/imgs/img{img_num}_glare{x+1}.png', img2[:,:,:3])
        # cv2.imwrite(f'./red/img{img_num}_glare{x+1}_mask.png', (img2 - img)[:, :, 2])
        # cv2.imwrite(f'./green/img{img_num}_glare{x+1}_mask.png', (img2 - img)[:, :, 1])
        # cv2.imwrite(f'./blue/img{img_num}_glare{x+1}_mask.png', (img2 - img)[:, :, 0])
        cv2.imwrite(f'../data/masks/img{img_num}_glare{x+1}_mask.png', (img2 - img)[:,:,:3])

    # # Adding glare3 to the image

    for x in range(6, 9):
        img3 = img.copy()
        glare3_copy = cv2.resize(glare3, (img.shape[1], img.shape[0]))

        height, width = img.shape[:2]
        tx = np.random.randint(img.shape[1] * -0.4, img.shape[1] * 0.4)
        ty = np.random.randint(img.shape[0] * -0.2, img.shape[0] * 0.1)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        glare3_copy = cv2.warpAffine(glare3_copy, translation_matrix, (width, height))

        alpha_background = img3[:, :, 3] / 255.0
        alpha_foreground = glare3_copy[:, :, 3] / 255.0

        for color in range(3):
            img3[:,:,color] = alpha_foreground * glare3_copy[:,:,color] + \
            alpha_background * img3[:,:,color] * (1 - alpha_foreground)

        img3[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
        cv2.imwrite(f'../data/imgs/img{img_num}_glare{x+1}.png', img3[:,:,:3])
        # cv2.imwrite(f'./red/img{img_num}_glare{x+1}_mask.png', (img3 - img)[:, :, 2])
        # cv2.imwrite(f'./green/img{img_num}_glare{x+1}_mask.png', (img3 - img)[:, :, 1])
        # cv2.imwrite(f'./blue/img{img_num}_glare{x+1}_mask.png', (img3 - img)[:, :, 0])
        cv2.imwrite(f'../data/masks/img{img_num}_glare{x+1}_mask.png', (img3 - img)[:,:,:3])

    # cv2.imwrite(f'./orig/img{img_num}.png', img[:,:,:3])


if __name__ == "__main__":
    asyncio.run(main())