from flask import Flask, render_template, redirect, request
import Caption_Image

#__name__==__main__
app=Flask(__name__)


@app.route('/')
def helloworld():
	# return render_template("index.html",User_Name_List=User_Name_List, count=count)
	return render_template("index.html")



# @app.route('/home')
# def home():
# 	return redirect('/')

@app.route('/', methods=['POST'])
def submit_data():
	if request.method=='POST':
		f=request.files['imagefile']
		path="./static/{}".format(f.filename)
		f.save(path)

		caption=Caption_Image.caption_the_image(path)

		result_dic={
		'image':path,
		'capton':caption
		}

		print(caption)
		print(f)

	return render_template("index.html",your_result=result_dic,your_caption=caption)


if __name__=='__main__':
	app.run(debug=False,threaded=False )
