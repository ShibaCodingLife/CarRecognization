package com.neusoft.neumooc.controller;

import java.util.HashMap;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import com.neusoft.core.common.BaseController;
import com.neusoft.core.common.conf.Params;
import com.neusoft.core.util.MD5Util;
import com.neusoft.core.util.Util;
import com.neusoft.neumooc.service.UserService;
import com.neusoft.neumooc.vo.UserVO;

/**
 * @ClassName: LoginController
 * @Description: 登录控制类
 */
@Controller
@RequestMapping(value = "/login")
public class LoginController extends BaseController {

	// 用户管理业务逻辑处理类
	@Autowired
	UserService userService;

	/**
	 * @Title: index
	 * @Description: 进入登录页
	 * @return 登录页
	 */
	@RequestMapping(value = "/index")
	public String index() {
		return "login/login";
	}

	/**
	 * @Title: login
	 * @Description: 登录
	 * @param session
	 *            HttpSession
	 * @param request
	 *            用户名、密码
	 * @return 是、否
	 */
	@RequestMapping(value = "/checkLogin")
	@ResponseBody
	public Map<String, Object> login(HttpSession session,
			HttpServletRequest request) {
		Map<String, Object> retMap = new HashMap<String, Object>();
		try {
			// 获取参数
			String userName = Util.notNull(request.getParameter("userName"));
			String password = Util.notNull(request.getParameter("password"));
			String passwordMD5 = MD5Util.gen(password.getBytes());

			// 用户信息
			UserVO user = userService.getUserByUserName(userName);

			// 用户名或密码是否正确
			if (user != null && passwordMD5.equals(user.getPassword())) {

				// 设置session
				session.setAttribute(Params.SESSION_ATTR_KEY_USER, user);

				// 封装返回值
				retMap.put(Params.RET_CODE, Params.RET_CODE_SUCCESS);
				retMap.put(Params.RET_OBJ, user);
				return retMap;
			} else {
				// 封装返回值 用户名或密码错误
				retMap.put(Params.RET_CODE, Params.RET_CODE_ERROR);
				retMap.put(Params.RET_OBJ, null);
				return retMap;
			}
		} catch (Exception e) {
			log.debug("【登录】登录出现异常" + e);
			// 封装返回值 异常
			retMap.put(Params.RET_CODE, Params.RET_CODE_ERROR);
			retMap.put(Params.RET_OBJ, null);
			return retMap;
		}
	}

	/**
	 * @Title: logout
	 * @Description: 退出
	 * @param session
	 *            HttpSession
	 * @return 登录页
	 */
	@RequestMapping(value = "/logout")
	public String logout(HttpSession session) {
		// 注销session
		session.invalidate();
		log.debug("【登录】注销session成功。");
		return "login/login";
	}

}
