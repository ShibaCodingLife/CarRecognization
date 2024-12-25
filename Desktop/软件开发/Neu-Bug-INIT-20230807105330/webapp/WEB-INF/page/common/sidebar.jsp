<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib uri="http://java.sun.com/jsp/jstl/functions" prefix="fn"%>
<%@ page import="org.springframework.web.util.UrlPathHelper"%>

<aside id="left-panel">
	<div class="login-info">
		<span> <a href="javascript:void(0);" id="show-shortcut"
			data-action="toggleShortcut"> <span></span>
		</a>
		</span>
	</div>
	<c:set var="uri"
		value="<%=new UrlPathHelper().getOriginatingRequestUri(request)%>" />
	<c:set var="contextPath"
		value="<%=new UrlPathHelper().getOriginatingContextPath(request)%>" />
	<input type="hidden" name="opMenu" id="opMenu" value="${uri}" />
	<nav>
		<ul>
			<li><a href="#"><i class="fa fa-lg fa-fw fa-delicious"></i>
					<span class="menu-item-parent">我的面板</span></a>
				<ul>
					<li
						class="<c:if test="${fn:startsWith(fn:substringAfter(uri, contextPath), '/panel/userInfo')}">active</c:if>"><a
						href="${base}/panel/userInfo">用户信息</a></li>
					<li
						class="<c:if test="${fn:startsWith(fn:substringAfter(uri, contextPath), '/panel/userPassword')}">active</c:if>"><a
						href="${base}/panel/userPassword">修改密码</a></li>
				</ul></li>
			<c:if
				test="${!empty sessionScope.user && (sessionScope.user.roleId=='1' || sessionScope.user.prjCount>0)}">
				<li><a href="#"><i class="fa fa-lg fa-fw fa-codepen"></i> <span
						class="menu-item-parent">项目管理</span></a>
					<ul>
						<li
							class="<c:if test="${fn:startsWith(fn:substringAfter(uri, contextPath), '/project/')}">active</c:if>"><a
							href="${base}/project/showList">项目管理</a></li>
					</ul></li>
			</c:if>
			<c:if
				test="${!empty sessionScope.user && (sessionScope.user.roleId=='1' || sessionScope.user.prjCount>0)}">
				<li><a href="#"><i class="fa fa-lg fa-fw fa-hand-o-right"></i>
						<span class="menu-item-parent">任务分配</span></a>
					<ul>
						<li
							class="<c:if test="${fn:startsWith(fn:substringAfter(uri, contextPath), '/funcAssign/')}">active</c:if>"><a
							href="${base}/funcAssign/showProjectList">任务分配</a></li>
					</ul></li>
			</c:if>
			<li><a href="#"><i class="fa fa-lg fa-fw fa-bug"></i> <span
					class="menu-item-parent">bug管理</span></a>
				<ul>
					<li
						class="<c:if test="${fn:startsWith(fn:substringAfter(uri, contextPath), '/bug/')}">active</c:if>"><a
						href="${base}/bug/showProjectList">bug管理</a></li>
				</ul></li>
			<c:if
				test="${!empty sessionScope.user && sessionScope.user.roleId=='1'}">
				<li><a href="#"><i class="fa fa-lg fa-fw fa-user"></i> <span
						class="menu-item-parent">用户管理</span></a>
					<ul>
						<li
							class="<c:if test="${fn:startsWith(fn:substringAfter(uri, contextPath), '/user/')}">active</c:if>"><a
							href="${base}/user/showList">用户管理</a></li>
					</ul></li>
			</c:if>
		</ul>
	</nav>
	<span class="minifyme" data-action="minifyMenu"> <i
		class="fa fa-arrow-circle-left hit"></i>
	</span>
</aside>
