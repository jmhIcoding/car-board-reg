__author__ = 'jmh081701'
from PIL import  Image
def img2mat(img_filename):
#把所有的图片都resize为20x20
    img = Image.open(img_filename)
    img = img.resize((20,20))
    mat = [[img.getpixel((x,y)) for x in range(0,img.size[0])] for y in range(0,img.size[1])]
    return mat
def test():
    mat = img2mat("dataset\\test\\1.bmp")
    print(mat)
    print(mat[0][0],len(mat),len(mat[0]))
if __name__ == '__main__':
    test()