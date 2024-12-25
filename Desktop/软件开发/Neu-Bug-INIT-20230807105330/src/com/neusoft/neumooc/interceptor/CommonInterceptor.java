package com.neusoft.neumooc.interceptor;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.springframework.web.servlet.handler.HandlerInterceptorAdapter;

import com.neusoft.core.common.conf.Params;

/**
 * @ClassName: CommonInterceptor
 * @Description: 访问需要登录的路径拦截
 */
public class CommonInterceptor extends HandlerInterceptorAdapter {

	/**
	 * @Fields mappingURL : 利用正则映射到需要拦截的路径
	 */
	@SuppressWarnings("unused")
	private String mappingURL;

	/**
	 * @Title: setMappingURL
	 * @Description: 设置需要拦截的路径
	 * @param mappingURL
	 */
	public void setMappingURL(String mappingURL) {
		this.mappingURL = mappingURL;
	}

	/**
	 * @Title: preHandle
	 * @Description: 访问拦截
	 * @param request
	 *            HttpServletRequest
	 * @param response
	 *            HttpServletResponse
	 * @param handler
	 *            Object
	 * @return boolean
	 * @throws Exception
	 */
	@Override
	public boolean preHandle(HttpServletRequest request,
			HttpServletResponse response, Object handler) throws Exception {

		// session
		HttpSession session = request.getSession();

		// 是否已登录
		if (session.getAttribute(Params.SESSION_ATTR_KEY_USER) != null) {
			return true;
		} else {
			request.getRequestDispatcher("/login/index").forward(request,
					response);
			return false;
		}
	}
}
