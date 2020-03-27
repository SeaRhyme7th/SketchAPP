let json_info = document.getElementById("json_info").value;
let show_img_pannel = document.getElementById("show_img_pannel")

json_info = JSON.parse(json_info);

for(var i = 0; i < 18; i++){
    var obj_li = document.createElement("li");
    obj_li.id = "li" + i;
    var obj_img = '<img src="' + 'static/' + json_info[i].path + '" alt="' + json_info[i].name + '" />';
    obj_li.innerHTML = '<a herf="">' + obj_img + '</a>' + '<span>' + json_info[i].name + '</span>';
    show_img_pannel.appendChild(obj_li);
}

var index = 0;

window.onload = function () {/*当页面加载完成后再执行这部分js代码*/
            var info = document.getElementById("info");
            info.innerHTML = (index+1)+"/5";
        }

function getNextImg() {

    var info = document.getElementById("info");
    index++;

    if(index > 4){  /*当图片已经翻到最后一张时，跳转到第一张图片*/
        index = 0;
    }

    for(var i = 0; i < 18; i++){
        var ii = i + index*18
        var obj_li = document.getElementById("li" + i);
        var obj_img = '<img src="' + 'static/' + json_info[ii].path + '" alt="' + json_info[ii].name + '" />';
        obj_li.innerHTML = '<a herf="">' + obj_img + '</a>' + '<span>' + json_info[ii].name + '</span>';
    }

    info.innerHTML = (index+1)+"/5";
}

function getProImg() {

    var info = document.getElementById("info");
    index--;

    if(index < 0){            /*当图片已经翻到第张时，跳转到最后一张图片*/
        index = 4;
    }

    for(var i = 0; i < 18; i++){
        var ii = i + index*18
        var obj_li = document.getElementById("li" + i);
        var obj_img = '<img src="' + 'static/' + json_info[ii].path + '" alt="' + json_info[ii].name + '" />';
        obj_li.innerHTML = '<a herf="">' + obj_img + '</a>' + '<span>' + json_info[ii].name + '</span>';
    }

    info.innerHTML = (index+1)+"/5";
}