/**
 * 这里是平台全局JS方法区
 * 各种公用的内容均放在此处，统一调用
 */
 
//输出 页头菜单,未登录显示
function outHeaderNoLogin(){
	outHeaderNoLogin('','');
}
function outHeaderNoLogin(dir, item){
	var rAddr = '';
	if(dir=='subDir'){
		rAddr = '../';
	} else {
		rAddr = dir;
	}
	
	var curCourse = "";
	var curProject = "";
	var curAnswer = "";
	if (item == "course") {
		curCourse = "active";
	} else if(item == "project") {
		curProject = "active";
	} else if(item == "answer") {
		curAnswer = "active";
	}
	
	document.write('<div class="navbar navbar-default navbar-fixed-top navbar-inverse" role="navigation">');
	document.write('		<div class="row bd-row">');
	document.write('		<div class="col-md-1 col-lg-1"></div>');
	document.write('		<div class="col-xs-12 col-sm-12 col-md-10 col-lg-10">');
	document.write('			<div class="navbar-header">');
	document.write('				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#index-navbar-collapse">');
	document.write('					<span class="sr-only">nav</span>');
	document.write('					<span class="icon-bar"></span>');
	document.write('					<span class="icon-bar"></span>');
	document.write('					<span class="icon-bar"></span>');
	document.write('				</button>');
	document.write('				<a class="navbar-brand" href="'+rAddr+'index/index.html" style="padding-top:0px;padding-left:0px;"><img src="'+rAddr+'img/logo.jpg" style="height:50px;"></img></a>');
	document.write('			</div>');
	document.write('			<div class="collapse navbar-collapse" id="index-navbar-collapse">');
	document.write('				<ul class="nav navbar-nav" style="width:auto;">');
	document.write('					<li class="'+curCourse+' nav-width"><a href="'+rAddr+'course/list_all.html" class="font-size-20 text-center">课程</a></li>');
	document.write('					<li class="'+curProject+' nav-width"><a href="'+rAddr+'project/list_all.html" class="font-size-20 text-center">项目</a></li>');
	document.write('					<li class="'+curAnswer+' nav-width"><a href="'+rAddr+'answer/answer_index.html" class="font-size-20 text-center">答疑</a></li>');
	document.write('				</ul>');
	document.write('				<ul class="nav navbar-nav navbar-right" style="width:auto;padding-right:7px;">');
	
	document.write('					<li>');
	document.write('						<a href="'+rAddr+'index/systemNotice.html" title="有系统新公告">');
	document.write('							<span class="label label-success">News &nbsp;');
	document.write('								<span class="badge" style="background-color:#fff;color:green;">12</span>');
	document.write('							</span>');
	document.write('						</a>');
	document.write('					</li>');	
	document.write('					<li><a href="#" class="font-size-16">登录</a></li>');
	document.write('					<li><a href="#" class="font-size-16">注册</a></li>');
	document.write('				</ul>');
	document.write('			</div>');
	document.write('		</div>');
	document.write('		<div class="col-md-1 col-lg-1"></div>');
	document.write('		</div>');
	document.write('	</div>');	
}

//输出 页头菜单,登录后的显示
function outHeaderLogined(){
	outHeaderLogined('','');
}

function outHeaderLogined(dir, item){
	var rAddr = '';
	if(dir=='subDir'){
		rAddr = '../';
	} else {
		rAddr = dir;
	}
	
	var curCourse = "";
	var curProject = "";
	var curAnswer = "";
	var curPersonal = "";
	if (item == "course") {
		curCourse = "active";
	} else if(item == "project") {
		curProject = "active";
	} else if(item == "answer") {
		curAnswer = "active";
	} else if(item == "personal") {
		curPersonal = "active";
	}
	
	document.write('<div class="navbar navbar-default navbar-fixed-top navbar-inverse" role="navigation">');
	document.write('		<div class="row bd-row">');
	document.write('		<div class="col-md-1 col-lg-1"></div>');
	document.write('		<div class="col-xs-12 col-sm-12 col-md-10 col-lg-10">');
	document.write('			<div class="navbar-header">');
	document.write('				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#index-navbar-collapse">');
	document.write('					<span class="sr-only">nav</span>');
	document.write('					<span class="icon-bar"></span>');
	document.write('					<span class="icon-bar"></span>');
	document.write('					<span class="icon-bar"></span>');
	document.write('				</button>');
	document.write('				<a class="navbar-brand" href="'+rAddr+'index/index.html" style="padding-top:0px;padding-left:0px;"><img src="'+rAddr+'img/logo.jpg" style="height:50px;"></img></a>');
	document.write('			</div>');
	document.write('			<div class="collapse navbar-collapse" id="index-navbar-collapse">');
	document.write('				<ul class="nav navbar-nav" style="width:auto;">');
	document.write('					<li class="'+curCourse+' nav-width"><a href="'+rAddr+'course/list_all.html" class="font-size-20 text-center">课程</a></li>');
	document.write('					<li class="'+curProject+' nav-width"><a href="'+rAddr+'project/list_all.html" class="font-size-20 text-center">项目</a></li>');
	document.write('					<li class="'+curAnswer+' nav-width"><a href="'+rAddr+'answer/answer_index.html" class="font-size-20 text-center">答疑</a></li>');
	document.write('					<li class="'+curPersonal+' nav-width"><a href="'+rAddr+'personal/dynamic.html" class="font-size-20 text-center">个人中心</a></li>');
	document.write('				</ul>');	
	document.write('				<ul class="header-dropdown-list hidden-xs" style="width:auto;">');	
	document.write('					<li><a href="#" class="white">Eileen</a></li>');
	document.write('					<li class="" id="overShowHead">');
	document.write('						<a href="#" class="dropdown-toggle headPhoto" data-toggle="dropdown" style="padding:6px 0px 4px 0px;margin-top:0px;"> ');
	document.write('							<img class="img-circle" src="'+rAddr+'img/photo_4.jpg" alt="" style="width:100%;height:100%;"></img>');
	document.write('							<b class="badge bounceIn animated">22</b>');
	document.write('						</a>');
	document.write('						<ul class="dropdown-menu pull-right" style="margin-top:0px;">');
	document.write('							<li><a href="../index/systemNotice.html" class="headPersonXL"><i class="fa fa-lg fa-fw fa-bullhorn stress-color"></i>&nbsp;系统公告<span class="badge pull-right inbox-badge">14</span></a></li>');
	document.write('							<li><a href="../personal/my_letters.html" class="headPersonXL"><i class="fa fa-lg fa-fw fa-envelope stress-color"></i>&nbsp;站内信<span class="badge pull-right inbox-badge">5</span></a></li>');
	document.write('							<li><a href="../personal/my_friend.html" class="headPersonXL"><i class="fa fa-lg fa-fw  fa-heart-o stress-color"></i>&nbsp;好友申请<span class="badge pull-right inbox-badge">3</span></a></li>');
	document.write('							<li class="divider"></li>');	
	document.write('							<li><a href="../personal/my_course.html" class="headPersonXL"><i class="fa fa-lg fa-fw fa-book stress-color"></i>&nbsp;我的课程</a></li>');
	document.write('							<li><a href="../personal/my_project.html" class="headPersonXL"><i class="fa fa-lg fa-fw fa-clipboard stress-color"></i>&nbsp;我的项目</a></li>');
	document.write('							<li><a href="../personal/growup_history.html" class="headPersonXL"><i class="fa fa-lg fa-fw fa-history stress-color"></i>&nbsp;成长历程</a></li>');
	document.write('							<li><a href="../personal/dynamic.html" class="headPersonXL"><i class="fa fa-lg fa-fw fa-home stress-color"></i>&nbsp;个人中心</a></li>');
	document.write('							<li class="divider"></li>');
	document.write('							<li><a href="#"><i class="fa fa-lg fa-fw fa-sign-out" style="color:#888;"></i>退出</a></li>');							
	document.write('						</ul>');
	document.write('					</li>');
	document.write('				</ul>');
	document.write('			</div>');
	document.write('		</div>');
	document.write('		<div class="col-md-1 col-lg-1"></div>');
	document.write('		</div>');
	document.write('	</div>');
}

//输出 页尾
function outFooter(){
	outFooter('');
}
function outFooter(dir){
	var rAddr = '';
	if(dir=='subDir'){
		rAddr = '../';
	} else {
		rAddr = dir;
	}
	document.write('');
	document.write('	<div style="background-color:#000;padding-bottom:20px;padding-top:20px;color:#ccc;">');
	document.write('		<div class="row bd-row">');
	document.write('			<div class="col-md-1 col-lg-1"></div>');
	document.write('			<div class="col-xs-12 col-sm-12 col-md-10 col-lg-10"> ');
	document.write('				<footer>');
	document.write('					<p class="pull-right"><a href="###">关于我们</a></p>');
	document.write('					<p class="pull-right"><a href="###">服务条款</a>&nbsp;|&nbsp;</p>');
	document.write('					<p class="pull-right"><a href="###">意见反馈</a>&nbsp;|&nbsp;</p>');
	document.write('					<p>Copyright © 2014 Neusoft 东软辽ICP备 12025617号-9</p>');
	document.write('				</footer>');
	document.write('			</div>');
	document.write('			<div class="col-md-1 col-lg-1"></div>');
	document.write('		</div>');
	document.write('	</div>');
}
