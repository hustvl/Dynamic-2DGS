import cv2

import numpy as np

def zoom_image(image_path, region,region2, zoom_factor):
    # 读取图像
    image = cv2.imread(image_path)

    # 提取放大区域的坐标
    x, y, width, height = region

    # 获取放大区域
    zoomed_region = image[y:y+height, x:x+width]

    # 计算放大后的区域尺寸
    zoomed_width = int(width * zoom_factor)
    zoomed_height = int(height * zoom_factor)

    # 使用指定的放大因子调整放大区域的尺寸
    zoomed_region = cv2.resize(zoomed_region, (zoomed_width, zoomed_height))

    # 在放大区域周围绘制边框
    cv2.rectangle(image, (x, y), (x+width, y+height), (100, 100, 0), 2)

    # 将放大后的区域放置在图像的左下角
    '''
    y_pos = image.shape[0] - zoomed_height
    image[y_pos-100:image.shape[0]-100, 0+10:zoomed_width+10] = zoomed_region

    # 在放置区域周围绘制边框
    cv2.rectangle(image, (0+10, y_pos-100), (zoomed_width+10, image.shape[0]-100), (100, 100, 0), 2)
    '''
    
    # 将放大后的区域放置在图像的右下角
    x_pos = image.shape[1] - zoomed_width
    y_pos = image.shape[0] - zoomed_height
    #image[y_pos:image.shape[0], x_pos:image.shape[1]] = zoomed_region
    #hook +10
    #pianyi = 30 #hook
    pianyi_l = 5 #hook
    pianyi_w = 25

    #image[0+40+pianyi_l:zoomed_height+40+pianyi_l, x_pos-pianyi_w:image.shape[1]-pianyi_w] = zoomed_region
    #cv2.rectangle(image, (image.shape[1]-pianyi_w, 0+40+pianyi_l), (image.shape[1]-zoomed_width-pianyi_w, zoomed_height+40+pianyi_l), (100, 100, 0), 2)
    image[x_pos-100+20:image.shape[1]-100+20,0+20+50:zoomed_height+20+50 ] = zoomed_region
    cv2.rectangle(image, (0+20+50, x_pos-100+20), (zoomed_height+20+50, image.shape[1]-100+20), (100, 100, 0), 2)
    
    
    # 提取放大区域的坐标
    x2, y2, width2, height2 = region2
    
    # 获取放大区域
    zoomed_region2 = image[y2:y2+height2, x2:x2+width2]

    # 计算放大后的区域尺寸
    zoomed_width2 = int(width2 * zoom_factor)
    zoomed_height2 = int(height2 * zoom_factor)

    # 使用指定的放大因子调整放大区域的尺寸
    zoomed_region2 = cv2.resize(zoomed_region2, (zoomed_width2, zoomed_height2))

    # 在放大区域周围绘制边框
    cv2.rectangle(image, (x2, y2), (x2+width2, y2+height2), (100, 100, 0), 2)

    # 将放大后的区域放置在图像的左下角
    
    #image[0+380+pianyi_l:zoomed_height+380+pianyi_l, x_pos-pianyi_w:image.shape[1]-pianyi_w] = zoomed_region2

    # 在放置区域周围绘制边框
    #cv2.rectangle(image, (image.shape[1]-pianyi_w, 0+380+pianyi_l), (image.shape[1]-zoomed_width-pianyi_w, zoomed_height+380+pianyi_l), (100, 100, 0), 2)
    
    image[x_pos-100+20:image.shape[1]-100+20,0+20+400:zoomed_height+20+400 ] = zoomed_region2
    cv2.rectangle(image, (0+20+400, x_pos-100+20), (zoomed_height+20+400, image.shape[1]-100+20), (100, 100, 0), 2)

    # 保存修改后的图像
    output_path = image_path.replace('.png', '_zoomed.png')

    #image2 = image[130:image.shape[0]-40,0:image.shape[1]-175]
    #others 
    image2 = image[15:-75,:]
    #hook
    #image2 = image[50:-40,:]
    #hellwarrior
    #image2 = image[30:-60,:]
    #image2 = image
    cv2.imwrite(output_path, image2)

    print(f"放大后的图像已保存为：{output_path}")
    return image2


def extend_left(image_path, extension_width, output_path):
    # 加载图像
    image = cv2.imread(image_path)

    # 创建新图像
    new_width = image.shape[1] + extension_width
    new_image = np.zeros((image.shape[0], new_width, 3), dtype=np.uint8)
    new_image.fill(255)

    # 将原始图像复制到新图像中
    new_image[:, :-extension_width] = image
    new_image = new_image[:, extension_width:]

    # 保存新图像
    cv2.imwrite(output_path, new_image)


def extend_down(image_path, extension_width, output_path):
    # 加载图像
    image = cv2.imread(image_path)

    # 创建新图像
    new_length = image.shape[0] + extension_width
    new_image = np.zeros((new_length,image.shape[1], 3), dtype=np.uint8)
    new_image.fill(255)

    # 将原始图像复制到新图像中
    new_image[ :-extension_width,:] = image
    new_image = new_image[extension_width:,:]

    # 保存新图像
    cv2.imwrite(output_path, new_image)
    
    
def caijian(image_path, output_path):
    # 加载图像
    image = cv2.imread(image_path)

    #new_image = image[70:-20, :]
    new_image = image[60:-120, :]

    # 保存新图像
    cv2.imwrite(output_path, new_image)

def margeimage(dvg,tiv,dgs3,dgs4,ours,gt,output_path):
    width1 = gt.shape[1]
    heigth1 = gt.shape[0]
    all_width = 3*width1
    all_heigth = 2*heigth1
    new_image = np.zeros((all_heigth, all_width, 3), dtype=np.uint8)
    new_image.fill(255)
    new_image[:heigth1,:width1] = dvg
    new_image[:heigth1,width1:2*width1] = tiv
    new_image[:heigth1,2*width1:3*width1] = dgs3
    new_image[heigth1:,:width1] = dgs4
    new_image[heigth1:,width1:2*width1] = ours
    new_image[heigth1:,2*width1:3*width1] = gt

    cv2.imwrite(output_path, new_image)




'''
for i in ['gt','3dgs','4dgs','dvg','tiv','ours']:
    # 示例用法
    image_path = f'/data5/zhangshuai/gaussian-splatting6/utils/066/{i}.png'  # 输入图像的路径
    image_path2 = f'/data5/zhangshuai/gaussian-splatting6/utils/066/{i}_extend.png'  # 输入图像的路径
    output_path3 = f'/data5/zhangshuai/gaussian-splatting6/utils/066/{i}_all.png'
    # region = (350, 400, 100, 100)   指定要放大的区域的坐标和尺寸 (x, y, width, height) 99
    #region = (500, 350, 100, 100) 
    #region2 = (800, 250, 100, 100) 

    region = (520, 350, 90, 90) 
    region2 = (720, 250, 90, 90) 
    zoom_factor = 4 # 放大因子


    extend_left(image_path,220,image_path2)
    if(i == 'gt'):
        gt = zoom_image(image_path2, region,region2, zoom_factor)
    elif(i == '3dgs'):
        dgs3 = zoom_image(image_path2, region,region2, zoom_factor)
    elif(i == '4dgs'):
        dgs4 = zoom_image(image_path2, region,region2, zoom_factor)
    elif(i == 'dvg'):
        dvg = zoom_image(image_path2, region,region2, zoom_factor)
    elif(i == 'tiv'):
        tiv = zoom_image(image_path2, region,region2, zoom_factor)
    else:
        ours = zoom_image(image_path2, region,region2, zoom_factor)


margeimage(dvg,tiv,dgs3,dgs4,ours,gt,output_path3)
'''

#jumpingjacks
'''
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/jumpingjacks_result/'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'{j}/{i}/0001.png'
            ours_i_path2 = output_path+f'{j}/{i}/v00001.png'
        elif i== 'gt':
            ours_i_path = output_path+f'{i}/00001.png'
            ours_i_path2 = output_path+f'{i}/v00001.png'
        else:
            ours_i_path = output_path+f'{j}/{i}/00001.png'
            ours_i_path2 = output_path+f'{j}/{i}/v00001.png'

        region = (240, 180, 80, 80) 
        region2 = (320, 395, 80, 80) 
        zoom_factor = 4 # 放大因子
        extend_left(ours_i_path,160,ours_i_path2)
        ours = zoom_image(ours_i_path2, region,region2, zoom_factor)
'''

'''
#standup
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/standup_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0016.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00016.png'
        elif i== 'gt':
            ours_i_path = output_path+f'/{i}/00016.png'
            ours_i_path2 = output_path+f'/{i}/v00016.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00016.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00016.png'

        region = (225, 173, 80, 80) 
        region2 = (255, 620, 80, 80) 
        zoom_factor = 4 # 放大因子
        extend_left(ours_i_path,160,ours_i_path2)
        ours = zoom_image(ours_i_path2, region,region2, zoom_factor)
'''

'''
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/bouncingballs_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0012.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00012.png'
        elif i== 'gt':
            ours_i_path = output_path+f'/{i}/00012.png'
            ours_i_path2 = output_path+f'/{i}/v00012.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00012.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00012.png'


        caijian(ours_i_path,ours_i_path2)
'''

'''
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/hook_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0002.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00002.png'
        elif i== 'gt':
            ours_i_path = output_path+f'/{i}/00002.png'
            ours_i_path2 = output_path+f'/{i}/v00002.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00002.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00002.png'

        region = (150, 340, 80, 80) 
        region2 = (245, 480, 80, 80) 
        zoom_factor = 4 # 放大因子
        extend_left(ours_i_path,160,ours_i_path2)
        ours = zoom_image(ours_i_path2, region,region2, zoom_factor)
'''

'''
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt','gt_mesh']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/beagle_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0021.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00021.png'
        elif i== 'gt' or i== 'gt_mesh':
            ours_i_path = output_path+f'/{i}/00021.png'
            ours_i_path2 = output_path+f'/{i}/v00021.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00021.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00021.png'


        caijian(ours_i_path,ours_i_path2)
        #extend_left(ours_i_path,150,ours_i_path2)

'''
'''
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/hellwarrior_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0002.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00002.png'
        elif i== 'gt':
            ours_i_path = output_path+f'/{i}/00002.png'
            ours_i_path2 = output_path+f'/{i}/v00002.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00002.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00002.png'

        region = (265, 180, 80, 80)
        region2 = (230, 270, 80, 80) 
        zoom_factor = 4 # 放大因子
        extend_left(ours_i_path,160,ours_i_path2)
        ours = zoom_image(ours_i_path2, region,region2, zoom_factor)

'''

'''
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/mutant_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0017.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00017.png'
        elif i== 'gt':
            ours_i_path = output_path+f'/{i}/00017.png'
            ours_i_path2 = output_path+f'/{i}/v00017.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00017.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00017.png'

        region = (325, 180, 80, 80)
        region2 = (240, 430, 80, 80) 
        zoom_factor = 4 # 放大因子
        extend_left(ours_i_path,185,ours_i_path2)
        ours = zoom_image(ours_i_path2, region,region2, zoom_factor)
'''

'''
for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/trex_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0013.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00013.png'
        elif i== 'gt':
            ours_i_path = output_path+f'/{i}/00013.png'
            ours_i_path2 = output_path+f'/{i}/v00013.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00013.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00013.png'

        region = (355, 50, 80, 80)
        region2 = (210, 400, 80, 80) 
        zoom_factor = 4 # 放大因子
        extend_left(ours_i_path,185,ours_i_path2)
        ours = zoom_image(ours_i_path2, region,region2, zoom_factor)
'''

for j in ['mesh','image']:
    for i in ['d3dgs','dgmesh','ours','scgs','gt']:
        output_path = '/data3/zhangshuai/SC-2DGSv2/outputs/lego_result'
        
        if i == 'dgmesh':
            ours_i_path = output_path+f'/{j}/{i}/0018.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00018.png'
        elif i== 'gt':
            ours_i_path = output_path+f'/{i}/00018.png'
            ours_i_path2 = output_path+f'/{i}/v00018.png'
        else:
            ours_i_path = output_path+f'/{j}/{i}/00018.png'
            ours_i_path2 = output_path+f'/{j}/{i}/v00018.png'

        region = (270, 50, 80, 80)
        region2 = (380, 230, 80, 80) 
        zoom_factor = 4 # 放大因子
        extend_down(ours_i_path,165,ours_i_path2)
        ours = zoom_image(ours_i_path2, region,region2, zoom_factor)