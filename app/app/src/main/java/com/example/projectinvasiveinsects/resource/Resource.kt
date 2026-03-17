package com.example.projectinvasiveinsects.resource

import android.content.res.Resources

sealed class Resource<T>(
    val data: T? = null,
    val message: String? = null,
    val status: Boolean? = false
) {
    class Success<T>(data: T, status: Boolean): Resource<T>(data, null, status)
    class Loading<T>(data: T? = null): Resource<T>(data, null, null)
    class Error<T>(data: T?, message: String): Resource<T>(data, message, null)
}