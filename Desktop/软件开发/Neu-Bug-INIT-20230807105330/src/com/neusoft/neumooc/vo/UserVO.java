package com.neusoft.neumooc.vo;

import com.neusoft.core.common.BaseVO;

/**
 * @ClassName: UserVO
 * @Description: 用户信息类
 * @date 2016年7月13日
 */
public class UserVO extends BaseVO {

	/**
	 * @Fields userId : 用户ID
	 */
	private String userId;

	/**
	 * @Fields userName : 用户名
	 */
	private String userName;

	/**
	 * @Fields password : 用户密码
	 */
	private String password;

	/**
	 * @Fields roleId : 用户角色 0:普通用户 1:管理员
	 */
	private String roleId;

	/**
	 * @Fields realName : 真实姓名
	 */
	private String realName;

	/**
	 * @Fields email : 邮箱
	 */
	private String email;

	/**
	 * @Fields status :用户状态 0删除 1未删除
	 */
	private String status;

	/**
	 * @Fields prjCount :负责项目数
	 */
	private String prjCount;

	/**
	 * @return userId 返回用户ID
	 */
	public String getUserId() {
		return userId;
	}

	/**
	 * @param userId
	 *            设置用户ID
	 */
	public void setUserId(String userId) {
		this.userId = userId;
	}

	/**
	 * @return userName 返回用户名
	 */
	public String getUserName() {
		return userName;
	}

	/**
	 * @param userName
	 *            设置用户名
	 */
	public void setUserName(String userName) {
		this.userName = userName;
	}

	/**
	 * @return password 返回用户密码
	 */
	public String getPassword() {
		return password;
	}

	/**
	 * @param password
	 *            设置用户密码
	 */
	public void setPassword(String password) {
		this.password = password;
	}

	/**
	 * @return roleId 返回用户角色
	 */
	public String getRoleId() {
		return roleId;
	}

	/**
	 * @param roleId
	 *            设置用户角色
	 */
	public void setRoleId(String roleId) {
		this.roleId = roleId;
	}

	/**
	 * @return realName 返回真实姓名
	 */
	public String getRealName() {
		return realName;
	}

	/**
	 * @param realName
	 *            设置真实姓名
	 */
	public void setRealName(String realName) {
		this.realName = realName;
	}

	/**
	 * @return email 返回邮箱
	 */
	public String getEmail() {
		return email;
	}

	/**
	 * @param email
	 *            设置邮箱
	 */
	public void setEmail(String email) {
		this.email = email;
	}

	/**
	 * @return email 返回用户状态
	 */
	public String getStatus() {
		return status;
	}

	/**
	 * @param status
	 *            设置用户状态
	 */
	public void setStatus(String status) {
		this.status = status;
	}

	/**
	 * @return prjCount 返回负责项目数
	 */
	public String getPrjCount() {
		return prjCount;
	}

	/**
	 * @param prjCount
	 *            设置负责项目数
	 */
	public void setPrjCount(String prjCount) {
		this.prjCount = prjCount;
	}

}
