<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>

<header id="header">
	<div id="logo-group">
		<span id="logo"><span
			style="margin-left: 20px; font-size: 18px;">软件缺陷管理系统</span></span>
	</div>
	<div class="pull-right">
		<!-- collapse menu button -->
		<div id="hide-menu" class="btn-header pull-right margin-right-10">
			<span> <a id="toggleMenu" href="javascript:void(0);"
				data-action="toggleMenu" data-placement="left" data-content="收缩菜单"><i
					class="fa fa-reorder"></i></a>
			</span>
		</div>
		<!-- end collapse menu -->
		<c:if test="${!empty sessionScope.user}">
			<!-- logout button -->
			<div id="logout" class="btn-header transparent pull-right">
				<span> <a href="${base}/login/logout" data-placement="left"
					data-content="退出"><i class="fa fa-sign-out"></i></a>
				</span>
			</div>
			<!-- end logout button -->

			<!-- search mobile button (this is hidden till mobile view port) -->
			<div class="btn-header transparent pull-right"
				style="margin-top: 20px;">
				<span id="sysUserId">欢迎您&nbsp;${sessionScope.user.realName}</span>
				&nbsp;&nbsp;&nbsp; <span class="hidden-xs" id="sysRoleId">角色:
					<c:if test="${sessionScope.user.roleId=='0'}">普通用户</c:if> <c:if
						test="${sessionScope.user.roleId=='1'}">管理员</c:if>
				</span>&nbsp;&nbsp;&nbsp;
			</div>
			<!-- end search mobile button -->
		</c:if>
	</div>
</header>
